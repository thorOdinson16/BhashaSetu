"""Bhasha Setu — Pipeline with ffmpeg audio conversion + parallel LLM entity extraction."""

import os
import json
import time
import asyncio
import tempfile
import subprocess
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.play import save

load_dotenv()

executor = ThreadPoolExecutor(max_workers=6)

ENTITY_PROMPT = """Extract entities from this transcript as structured JSON.
Return ONLY valid JSON, no other text. No markdown fences.

Transcript: "{transcript}"

Keys (use null if not present):
- date: final agreed date after any correction
- time: time mentioned
- name: person name
- phone: phone number
- address: address or location
- amount: monetary amount
- correction_detected: true if self-correction was found
- correction_old_value: original value before correction, or null
- correction_new_value: corrected final value, or null"""

RECEIPT_PROMPT = """Generate a bilingual receipt in Kannada and Hindi from these entities.
Return ONLY valid JSON, no other text. No markdown fences.

Entities: {entities}

JSON format:
{{
  "entities": [
    {{
      "type": "date|time|name|phone|address|amount",
      "value": "final value in English",
      "kn": "value in Kannada",
      "hi": "value in Hindi",
      "correction_detected": false,
      "correction_old_value": null,
      "correction_new_value": null
    }}
  ]
}}"""

RECEIPT_PROMPT = """Generate a bilingual receipt in Kannada and Hindi from these entities.
Return ONLY valid JSON, no other text. No markdown fences.

Entities: {entities}

JSON format:
{{
  "entities": [
    {{
      "type": "date|time|name|phone|address|amount",
      "value": "final value in English",
      "kn": "value in Kannada",
      "hi": "value in Hindi",
      "correction_detected": false,
      "correction_old_value": null,
      "correction_new_value": null
    }}
  ]
}}"""


def _convert_audio(raw_bytes: bytes) -> bytes:
    """Convert non-WAV audio (WebM, OGG, MP3) to 16kHz mono WAV via ffmpeg.
    Strips metadata and caps at 10s to avoid duration issues."""
    if raw_bytes[:4] == b"RIFF":
        return raw_bytes
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-map_metadata", "-1",
             "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
             "-t", "10", "-f", "wav", "pipe:1"],
            input=raw_bytes,
            capture_output=True,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout and len(proc.stdout) > 100:
            return proc.stdout
    except Exception:
        pass
    return raw_bytes


class Pipeline:
    def __init__(self):
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY not set in environment")
        self.client = SarvamAI(api_subscription_key=api_key)

    def _stt(self, audio_path: str) -> tuple[str, float, str | None]:
        """Transcribe audio via Saaras v3 REST."""
        t0 = time.time()
        try:
            with open(audio_path, "rb") as f:
                resp = self.client.speech_to_text.transcribe(
                    file=f, model="saaras:v3", mode="transcribe"
                )
            elapsed = time.time() - t0
            return resp.transcript, elapsed, None
        except Exception as e:
            elapsed = time.time() - t0
            return "", elapsed, str(e)

    def _translate(self, text: str, source: str, target: str) -> tuple[str, float, str | None]:
        """Translate text via Sarvam Translate REST."""
        t0 = time.time()
        try:
            resp = self.client.text.translate(
                input=text,
                source_language_code=source,
                target_language_code=target,
            )
            elapsed = time.time() - t0
            return resp.translated_text, elapsed, None
        except Exception as e:
            elapsed = time.time() - t0
            return "", elapsed, f"Translate failed: {e}"

    def _tts(self, text: str, lang: str, speaker: str = "shubh") -> tuple[str | None, float, str | None]:
        """Generate speech via Bulbul v3 REST. Returns (audio_path, elapsed, error)."""
        t0 = time.time()
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                target_language_code=lang,
                model="bulbul:v3",
                speaker=speaker,
            )
            elapsed = time.time() - t0
            out_path = tempfile.mktemp(suffix=".wav")
            save(audio, out_path)
            return out_path, elapsed, None
        except Exception as e:
            elapsed = time.time() - t0
            return None, elapsed, f"TTS failed: {e}"

    def _extract_entities(self, transcript: str) -> dict:
        """Extract entities from transcript via Sarvam-105B."""
        t0 = time.time()
        try:
            prompt = ENTITY_PROMPT.format(transcript=transcript)
            resp = self.client.chat.completions(
                model="sarvam-105b",
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            content = resp.choices[0].message.content or ""
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            entities = json.loads(content)
            entities["_llm_latency"] = round(elapsed, 2)
            return entities
        except json.JSONDecodeError:
            return {"raw": content, "error": "JSON parse failed", "_llm_latency": round(time.time() - t0, 2)}
        except Exception as e:
            return {"error": str(type(e).__name__), "_llm_latency": round(time.time() - t0, 2)}

    def _generate_bilingual(self, entities: dict) -> list[dict]:
        """Generate Kannada and Hindi translations for entity values via Sarvam-105B."""
        try:
            prompt = RECEIPT_PROMPT.format(entities=json.dumps(entities, ensure_ascii=False))
            resp = self.client.chat.completions(
                model="sarvam-105b",
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content or ""
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(content).get("entities", [])
        except Exception:
            return []

    async def process_utterance(
        self, raw_bytes: bytes, source_lang: str, target_lang: str, speaker: str = "shubh"
    ) -> dict:
        """Full pipeline: STT → Translate → TTS + LLM (parallel)."""

        wall_start = time.time()

        # Always convert non-WAV audio via ffmpeg (browser WebM → 16kHz WAV)
        wav_bytes = await asyncio.get_event_loop().run_in_executor(
            executor, _convert_audio, raw_bytes
        )
        stt_path = tempfile.mktemp(suffix=".wav")
        Path(stt_path).write_bytes(wav_bytes)

        # ── STT ──
        transcript, stt_time, stt_err = await asyncio.get_event_loop().run_in_executor(
            executor, self._stt, stt_path
        )
        if stt_err:
            return {"error": stt_err, "transcript": transcript}

        # ── Translate ──
        translated, tr_time, tr_err = await asyncio.get_event_loop().run_in_executor(
            executor, self._translate, transcript, source_lang, target_lang
        )
        if tr_err:
            return {"error": tr_err, "transcript": transcript, "translated_text": translated}

        # ── TTS + LLM in parallel ──
        tts_future = asyncio.get_event_loop().run_in_executor(
            executor, self._tts, translated, target_lang, speaker
        )
        llm_future = asyncio.get_event_loop().run_in_executor(
            executor, self._extract_entities, transcript
        )

        (audio_path, tts_time, tts_err) = await tts_future
        llm_result = await llm_future
        wall_total = round(time.time() - wall_start, 2)

        audio_b64 = None
        if audio_path and Path(audio_path).exists():
            audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()

        return {
            "transcript": transcript,
            "translated_text": translated,
            "entities": llm_result,
            "audio_path": audio_path,
            "audio_b64": audio_b64,
            "timing": {
                "stt": round(stt_time, 2),
                "translate": round(tr_time, 2),
                "tts": round(tts_time, 2),
                "llm": llm_result.get("_llm_latency", 0),
                "total": wall_total,
            },
            "tts_error": tts_err,
        }

    def generate_receipt(self, entities_list: list[dict]) -> dict:
        """Generate a bilingual receipt from a list of entity dicts."""
        all_entities = []
        for ent in entities_list:
            if ent.get("error"):
                continue
            bilingual = self._generate_bilingual(ent)
            all_entities.extend(bilingual)
        return {
            "entities": all_entities,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def score_scenario(self, receipt: dict, ground_truth: dict) -> dict:
        """Score a receipt against ground truth using the 7-check rubric."""
        checks = {
            "date": False, "time": False, "name": False,
            "phone": False, "address": False, "amount": False,
            "correction": False,
        }
        ent_map = {}
        for e in receipt.get("entities", []):
            ent_map[e.get("type", "")] = e

        for field in ["date", "time", "name", "phone", "address", "amount"]:
            gt_val = ground_truth.get(field)
            rcpt_ent = ent_map.get(field, {})
            rcpt_val = rcpt_ent.get("value", "")
            if gt_val is None:
                checks[field] = rcpt_val == "" or rcpt_val is None or rcpt_val == "—"
            else:
                checks[field] = str(gt_val).lower() in str(rcpt_val).lower()

        checks["correction"] = (
            ground_truth.get("correction_detected") == bool(
                ent_map.get("date", {}).get("correction_detected")
                or ent_map.get("amount", {}).get("correction_detected")
            )
        )

        score = sum(1 for v in checks.values() if v)
        return {"checks": checks, "score": score, "total": 7, "pct": round(score / 7 * 100)}
