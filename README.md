PTZ-Detector

Cross-platform PTZ object detector (Windows/Linux).

[main repo](https://github.com/ArtemShaputko/PTZ-Detector)

## Install

- Install python deps: `python -m pip install -r requirements.txt`
- Download Vosk RU model (required for voice control):
  - [https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip)
  - unpack to project root so folder path is `./vosk-model-small-ru-0.22`

## Run

- Main Qt app: `python program/object_targeter/main_qt.py`
- Legacy headless pipeline: `python program/object_targeter/main.py`

## Platform notes

- **Linux**
  - Camera source defaults to `/dev/video0`.
  - If selected source is `/dev/videoX`, FFmpeg + `v4l2` backend is used.
- **Windows**
  - Camera source defaults to index `0`.
  - OpenCV `VideoCapture` backend is used for numeric camera indexes.

- Microphone list and selection use `sounddevice` on both platforms.
- Serial PTZ output auto-detects available COM/tty ports; if no serial port is found, app still runs with PTZ serial output disabled.
