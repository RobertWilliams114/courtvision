# CourtVision

**Ask a question about a basketball game in plain English; get the commentary that answers it, the video clips that prove it, and a court diagram showing where it happened.**

Commentary tells you *what* happened. Video shows you *that* it happened. CourtVision links the two: a question is answered from the broadcast audio, then the system jumps to those exact timestamps, extracts the clips, runs player detection on them, and renders bird's-eye court diagrams — so every claim comes with visual evidence.

Runs locally end to end. No cloud APIs.

---

## What it does

```
"What did LeBron do in the fourth quarter?"
        │
        ├─ Answer ── search commentary embeddings ─→ relevant segments + timestamps
        │                                                      │
        └─ Verify ── seek to those timestamps ────────────────┘
                       ├─ extract 10-second clips
                       ├─ YOLOv8 player detection
                       └─ bird's-eye court diagram
        │
   answer + clips + diagrams, together
```

**Commentary Q&A.** `faster-whisper` transcribes the broadcast audio with word-level timestamps. Segments are embedded and indexed in ChromaDB. A question retrieves the relevant segments, and a local LLM (`llama3.2` via Ollama) answers from them.

**Video verification.** The timestamps from the commentary search drive clip extraction with OpenCV. YOLOv8 detects players in the extracted frames, and their positions are projected onto a court schematic.

The integration is the point: retrieval and vision are not two separate demos, they are one query path.

---

## Results on the sample game

| | |
|---|---|
| Source | NBA broadcast, 94 min, 640×360 h264 |
| Transcribed | **597 commentary segments** across ~25 minutes of audio |
| Detection | YOLOv8 (`yolov8n`) |
| Retrieval | ChromaDB over segment embeddings |
| Generation | `llama3.2` via Ollama |

The notebook ships with its outputs intact, including the generated court diagrams, so you can read the results without running anything.

---

## Quickstart

**Requirements:** Python 3.10–3.13, [Ollama](https://ollama.com), an NVIDIA GPU strongly recommended for detection.

```bash
# 1. Model
ollama pull llama3.2

# 2. PyTorch FIRST — pick the CUDA build for your GPU
#    (Blackwell / RTX 50-series needs CUDA 12.8+)
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Everything else
uv pip install -r requirements.txt

# 4. Run
jupyter notebook courtvision.ipynb
```

Ask about a specific player — that's what triggers clip extraction and diagram generation.

### Getting the data

**Video and audio are not in this repository** — they're copyrighted broadcast footage, and they're large. Supply your own:

```bash
export COURTVISION_VIDEO=/path/to/your/game.mp4
```

Default path if unset is `video/basketball_video.mp4`.

A pre-computed transcript (`transcription.json`, 597 segments) **is** included, so the retrieval and Q&A pipeline runs immediately without re-transcribing or even having the audio. Only video verification needs the video.

---

## Stack

Python 3.12 · faster-whisper · ChromaDB · sentence-transformers · Ollama (`llama3.2`) · Ultralytics YOLOv8 · OpenCV · matplotlib

## Limitations

- Source footage is 640×360, which caps detection quality — players at the far end of the court are unreliable
- No player *identification*: detection finds people, but linking a detection to the name in the commentary is positional and approximate
- Bird's-eye projection uses a fixed court assumption rather than a calibrated homography, so positions are indicative, not measured
- Transcription accuracy degrades under crowd noise, which is most of a basketball broadcast
- Single-notebook pipeline; not yet refactored into an importable package

## Provenance

Built for CS 370 (Introduction to AI) at NJIT, Fall 2025, and cleaned up since. The pipeline design and the integration between the two halves are my own.
