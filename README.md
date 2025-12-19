# Assignment 4: Basketball Commentary Analysis & Video Verification

## Overview
This assignment implements a complete basketball game analysis system that combines:
- **Task 1**: Audio transcription and Q&A system using RAG (Retrieval-Augmented Generation)
- **Task 2**: Video verification with clip extraction and bird's eye view diagrams

## Files

### Required Files
- **assignment-4.ipynb** - Main notebook (run this for complete assignment)
- **game_portion.mp3** - Audio file for transcription (Task 1)
- **video/basketball_video.mp4** - Full game video for verification (Task 2)
- **yolov8n.pt** - YOLO model weights for player detection
- **transcription.json** - Pre-generated transcription with timestamps
- **transcription.txt** - Human-readable transcription

### Output Directory
- **task2_outputs/** - Generated video clips and bird's eye diagrams

## Prerequisites

I used ollama desktop because the docker container was giving me an error that I could not get past.

### 2. Python Dependencies
The notebook will install required packages:
- faster-whisper (audio transcription)
- chromadb (vector database)
- ollama (LLM interface)
- opencv-python (video processing)
- ultralytics (YOLO detection)
- matplotlib (diagrams)

## How to Run

1. **Open the notebook**: `assignment-4.ipynb`

**When in the notebook, please ask about specific players as this will produce the videos and bird eye view diagrams.**

2. **Run cells in order**:
   - Test Ollama Desktop connection
   - Run Task 1 (audio transcription → vector store → Q&A system)
   - Run Task 2 (video clip extraction → bird's eye diagrams)


## Integration Workflow

When you ask about a player:
1. **Task 1**: Searches commentary for player mentions
2. **Task 2**: Extracts video clips at those timestamps
3. **Task 2**: Generates bird's eye court diagrams
4. **Display**: Shows commentary answer + video clips + diagrams together

This will:
- Find all commentary segments mentioning LeBron
- Create 10-second video clips around each mention
- Generate court diagrams showing player position
- Display everything together for verification

## Outputs

After running, check `task2_outputs/` for:
- `player_actions.csv` - Tracked player actions
- `play_segmentations.csv` - Segmented plays
- `video_clips/` - Extracted video clips
- `diagrams/` - Bird's eye court diagrams

## Notes

- The notebook is self-contained and runs both Task 1 and Task 2
- No need to run any other notebooks
- Ollama Desktop runs locally on port 11434
- All outputs are saved to `task2_outputs/` directory
