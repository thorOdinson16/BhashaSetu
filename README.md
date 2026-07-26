# Bhasha Setu — Instant Language Bridge for Kannada & Hindi

Two people speaking Kannada and Hindi converse naturally, hear fast translations, and leave with one verified shared receipt where every date, amount, and name matches on both sides.

Built for **Sarvam Epoch Buildathon 2026** — solo, 6 hours.

---

## How It Works

```
Speaker A (Kannada)                         Speaker B (Hindi)
      │                                             │
      ▼                                             ▼
  Browser PCM → WAV                             Browser PCM → WAV
      │                                             │
      ▼                                             ▼
  Saaras v3 STT (~1.5s)                       Saaras v3 STT (~1.5s)
      │                                             │
      ▼                                             ▼
  Sarvam Translate (~0.7s)                    Sarvam Translate (~0.7s)
      │                                             │
      ├── REST TTS (~2.5s) ──► Hindi audio ──► Speaker B hears (~5s)
      │
      └── LLM (background, ~15s) ──► Entities + Enhanced translation + TTS
              │
              ▼
         WebSocket callback → Both tabs get entities + receipt
```

**Fast path (~5s):** STT → Translate → TTS relay. Listener hears the translation immediately.

**Enhanced path (~15s):** Sarvam-105B extracts entities, produces better translation, generates TTS. Arrives via WebSocket callback — never blocks the relay.

**Two-tab session:** Each browser tab connects via WebSocket. One tab speaks Kannada, the other Hindi. Audio recorded as raw PCM, encoded to WAV client-side — no ffmpeg, no duration issues.

---

## Features

- **Real-time two-person voice relay** — Kannada ↔ Hindi with measured ~5s latency
- **Entity extraction** — dates, times, names, phone numbers, addresses, amounts
- **Self-correction propagation** — "Thursday... no, Friday" updates both sides with strike-through
- **Shared canonical state** — one list of facts both parties agree on, not two disconnected transcripts
- **Bilingual receipt** — shareable URL with all confirmed entities visible in both languages
- **Conversation log** — complete turn history with transcripts and translations
- **Receipt log** — auto-generated receipt links for every exchange
- **Enhanced LLM translation** — arrives ~15s after fast relay with better quality + TTS audio
- **Upload testing page** — file-based testing at `/upload`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Speech-to-Text | Sarvam Saaras v3 |
| Translation | Sarvam Translate |
| Text-to-Speech | Sarvam Bulbul v3 (30+ voices) |
| LLM / Entity Extraction | Sarvam-105B |
| Backend | FastAPI + WebSocket |
| Frontend | Vanilla HTML/CSS/JS, dark theme |
| Audio | Browser AudioContext → raw PCM → WAV encoding |

---

## Quick Start

```bash
# 1. Clone and install
cd BhashaSetu
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set API key
echo "SARVAM_API_KEY=your_key_here" > .env

# 3. Start server
python server.py

# 4. Open in browser
# Tab 1: http://localhost:8000 (auto-generates session)
# Tab 2: Paste the URL from Tab 1 to join same session
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Live bridge page (auto session ID) |
| `/upload` | GET | File upload testing page |
| `/upload` | POST | Process audio file pipeline |
| `/ws/{session_id}` | WebSocket | Two-tab session connection |
| `/receipt/{id}` | GET | Shareable receipt page |
| `/api/receipt` | POST | Generate receipt from entities |
| `/api/receipts` | GET | List all receipt IDs |

---

## Demo Flow (2 minutes)

| Time | What |
|---|---|
| 0–10s | "A Hindi-speaking tenant calls a Kannada landlord. The tenant corrects the date mid-sentence. Without Bhasha Setu — the landlord only hears the first date. Wrong repair day." |
| 10–40s | Two participants speak in different languages via live bridge. Audio relay at ~5s. Correction visible with strike-through. |
| 40–60s | Enhanced translation arrives (LLM quality). Shared facts update. Receipt auto-generated with matching facts in both languages. |
| 60–90s | Open shareable receipt URL in new tab — same bilingual record, correction history preserved. |
| 90–120s | "Currently: third human interpreter. Bhasha Setu: one verified record in both languages." |

---

## Project Structure

```
BhashaSetu/
├── server.py              # FastAPI + WebSocket server
├── pipeline.py            # STT → Translate → TTS + parallel LLM
├── score_fixtures.py      # JTBD scoring rubric against ground truth
├── requirements.txt       # Python dependencies
├── .env                   # SARVAM_API_KEY (gitignored)
├── .gitignore
├── static/
│   ├── live.html          # Live two-tab bridge UI
│   └── index.html         # File upload testing UI
└── fixtures/
    ├── ground_truth.json  # Expected entities for scoring
    ├── test_kan.wav       # Test utterance
    ├── scenario_1_*.wav   # Landlord-tenant repair scheduling
    ├── scenario_2_*.wav   # Delivery agent-customer coordination
    ├── scenario_3_*.wav   # Coworkers splitting a bill
    └── scenario_4_*.wav   # Unseen meetup scheduling
```

---

## Known Limitations

- **Sarvam-105B latency:** Enhanced path takes 10–20s. Runs in background, never blocks relay.
- **REST only:** STT and TTS use REST endpoints. WebSocket streaming STT/TTS would reduce latency further.
- **Entity extraction accuracy:** Varies on complex code-mixed utterances. Improved by enhanced translation fallback.
- **Two-party only:** Built for two speakers.
- **Browser-only:** No telephony integration. Browser microphone capture via AudioContext.
- **Hindi & Kannada only:** Depth on one language pair beats breadth across many.
