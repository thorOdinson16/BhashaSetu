"""Bhasha Setu — FastAPI server with WebSocket session support."""

import os
import json
import uuid
import time
import asyncio
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import Pipeline

load_dotenv()

app = FastAPI(title="Bhasha Setu")
pipeline = Pipeline()
executor = ThreadPoolExecutor(max_workers=4)

receipts: dict[str, dict] = {}
sessions: dict[str, dict] = {}

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/live", response_class=HTMLResponse)
async def live():
    return (STATIC_DIR / "live.html").read_text(encoding="utf-8")


@app.get("/receipt/{receipt_id}", response_class=HTMLResponse)
async def view_receipt(receipt_id: str):
    receipt = receipts.get(receipt_id)
    if not receipt:
        return HTMLResponse("<h2>Receipt not found</h2>", status_code=404)
    return render_receipt_page(receipt_id, receipt)


@app.get("/api/receipts")
async def list_receipts():
    return JSONResponse(list(receipts.keys()))


@app.post("/api/receipt")
async def create_receipt(data: list[dict] | dict = None):
    entities = data if isinstance(data, list) else []
    if isinstance(data, dict):
        entities = data.get("entities", [])
    receipt_id = str(uuid.uuid4())[:8]
    receipt = pipeline.generate_receipt(entities)
    receipt["receipt_id"] = receipt_id
    receipts[receipt_id] = receipt
    return JSONResponse({"receipt_id": receipt_id, "receipt": receipt})


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/upload")
async def upload_and_relay(
    file: UploadFile = File(...),
    source_lang: str = Form("kn-IN"),
    target_lang: str = Form("hi-IN"),
    speaker: str = Form("shubh"),
    session_id: str = Form(""),
):
    audio_bytes = await file.read()

    # Build entity callback for this session
    on_entities = None
    if session_id and session_id in sessions:
        session = sessions[session_id]
        async def _broadcast_entities(entities, transcript, translated):
            for conn in session.get("connections", []):
                try:
                    await conn.send_json({
                        "type": "entities",
                        "entities": entities,
                        "transcript": transcript,
                        "translated": translated,
                    })
                except Exception:
                    pass
        on_entities = _broadcast_entities

    result = await pipeline.process_utterance(
        audio_bytes, source_lang, target_lang, speaker, on_entities=on_entities
    )

    if "error" in result:
        return JSONResponse(result, status_code=500)

    out_path = UPLOAD_DIR / f"out_{uuid.uuid4().hex[:8]}.wav"
    if result.get("audio_path") and Path(result["audio_path"]).exists():
        import shutil
        shutil.copy(result["audio_path"], out_path)
        result["audio_url"] = f"/audio/{out_path.name}"

    # Entities arrive later via WebSocket callback; skip receipt here
    result["entities"] = None

    return JSONResponse(result)


@app.websocket("/ws/{session_id}")
async def websocket_session(ws: WebSocket, session_id: str):
    await ws.accept()

    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "connections": [],
            "entities": [],
            "turn_history": [],
            "participants": {},
        }

    session = sessions[session_id]
    session["connections"].append(ws)
    participant = {"id": str(uuid.uuid4())[:6]}
    session["participants"][participant["id"]] = participant

    await ws.send_json({"type": "joined", "session_id": session_id, "participant_id": participant["id"]})

    # Notify other connections
    for conn in session["connections"]:
        if conn != ws:
            try:
                await conn.send_json({"type": "peer_joined", "count": len(session["connections"])})
            except Exception:
                pass

    try:
        while True:
            msg = await ws.receive_json()

            if msg.get("type") == "relay_broadcast":
                # Broadcast to all other connections in this session
                for conn in session["connections"]:
                    if conn != ws:
                        try:
                            await conn.send_json({
                                "type": "relay",
                                "transcript": msg.get("transcript", ""),
                                "translated": msg.get("translated", ""),
                                "audio_b64": msg.get("audio_b64"),
                                "entities": msg.get("entities", {}),
                            })
                        except Exception:
                            pass

            elif msg.get("type") == "set_language":
                participant["lang"] = msg.get("lang", "kn-IN")
                participant["speaker"] = msg.get("speaker", "shubh")

    except WebSocketDisconnect:
        pass
    finally:
        session["connections"].remove(ws)


async def handle_audio(ws, session, participant, msg):
    """Process audio bytes: STT → Translate → TTS, broadcast results."""
    audio_b64 = msg.get("data", "")
    if not audio_b64:
        return

    import base64
    audio_bytes = base64.b64decode(audio_b64)

    source_lang = participant.get("lang", "kn-IN")
    other_langs = {
        "kn-IN": "hi-IN",
        "hi-IN": "kn-IN",
    }
    target_lang = other_langs.get(source_lang, "hi-IN")
    speaker = participant.get("speaker", "shubh")

    # Run pipeline
    result = await pipeline.process_utterance(audio_bytes, source_lang, target_lang, speaker)

    if "error" in result:
        await ws.send_json({"type": "error", "message": result["error"]})
        return

    # Persist entity if extracted
    if result.get("entities") and not result["entities"].get("error"):
        ent = result["entities"]
        ent["_speaker"] = participant["id"]
        ent["_source_lang"] = source_lang
        session["entities"].append(ent)

    # Send result back to speaker
    speaker_msg = {
        "type": "result",
        "transcript": result.get("transcript", ""),
        "translated": result.get("translated_text", ""),
        "entities": result.get("entities", {}),
        "timing": result.get("timing", {}),
        "audio_b64": _audio_to_b64(result.get("audio_path")),
    }
    await ws.send_json(speaker_msg)

    # Broadcast to other participants
    broadcast_msg = {
        "type": "relay",
        "transcript": result.get("transcript", ""),
        "translated": result.get("translated_text", ""),
        "entities": result.get("entities", {}),
        "audio_b64": _audio_to_b64(result.get("audio_path")),
        "from_participant": participant["id"],
    }
    for conn in session["connections"]:
        if conn != ws:
            try:
                await conn.send_json(broadcast_msg)
            except Exception:
                pass


def _audio_to_b64(audio_path: str | None) -> str | None:
    """Read audio file and return base64 string."""
    if not audio_path or not Path(audio_path).exists():
        return None
    import base64
    return base64.b64encode(Path(audio_path).read_bytes()).decode()


def render_receipt_page(receipt_id: str, receipt: dict) -> str:
    entities = receipt.get("entities", [])
    rows = ""
    for e in entities:
        old = e.get("correction_old_value")
        new = e.get("correction_new_value")
        has_correction = e.get("correction_detected") and old and new
        if has_correction:
            value_cell = f"<s>{old}</s> → <strong>{new}</strong>"
        else:
            value_cell = e.get("value", "") or e.get("kn", "") or "—"

        kn_val = e.get("kn", "") or "—"
        hi_val = e.get("hi", "") or "—"

        rows += f"""<tr>
            <td>{e.get('type', '')}</td>
            <td>{value_cell}</td>
            <td>{kn_val}</td>
            <td>{hi_val}</td>
        </tr>"""

    source = receipt.get("source_transcript", "")
    translated = receipt.get("translated_text", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Receipt {receipt_id} — Bhasha Setu</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 1rem; background:#0f0f0f; color:#e0e0e0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 10px 14px; text-align: left; font-size: 14px; }}
  th {{ background: #1a1a1a; color: #aaa; font-weight: 600; }}
  s {{ color: #888; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
  .badge {{ background: #1b5e20; color: #4caf50; padding: 4px 14px; border-radius: 12px; font-size: 13px; }}
  h2 {{ color: #fff; margin: 0; }}
  h2 span {{ color: #4caf50; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 1.5rem; }}
  .box {{ background: #111; border: 1px solid #333; border-radius: 6px; padding: 12px; margin-bottom: 1rem; font-size: 13px; max-height: 80px; overflow-y: auto; }}
</style></head>
<body>
  <div class="header">
    <h2>Bhasha <span>Setu</span> Receipt</h2>
    <span class="badge">Confirmed</span>
  </div>
  <p class="subtitle">ID: {receipt_id} · {receipt.get('generated_at', '')}</p>
  <div class="box"><strong>Source:</strong> {source}</div>
  <div class="box"><strong>Translation:</strong> {translated}</div>
  <table>
    <tr><th>Entity</th><th>Value</th><th>Kannada</th><th>Hindi</th></tr>
    {rows}
  </table>
  <p style="margin-top: 1.5rem; color: #666; font-size: 13px;">
    Generated by Bhasha Setu — a language bridge for Kannada and Hindi speakers.
  </p>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
