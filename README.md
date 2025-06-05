# Real-Time Lip-Syncing WebSocket API

Brief project description: This API uses [Chosen AI Model Name, e.g., Wav2Lip] to generate a lip-synced video from an input audio file and a person's image. Communication is handled via WebSockets.

## Table of Contents
- [How it Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Downloading Pretrained Models](#downloading-pretrained-models)
- [Running with Docker](#running-with-docker)
- [API Usage](#api-usage)
  - [WebSocket Endpoint](#websocket-endpoint)
  - [Input Format](#input-format)
  - [Output Format](#output-format)
- [Testing with a WebSocket Client](#testing-with-a-websocket-client)

---

## How it Works

The system enables real-time lip-syncing through a WebSocket interface:
1.  A client establishes a WebSocket connection to the server at `/ws/lip-sync`.
2.  The client sends a JSON payload containing:
    * Base64 encoded audio (e.g., WAV).
    * Base64 encoded image of a person (e.g., JPG, PNG).
    * Parameters for processing like FPS, padding, image size, etc.
3.  The FastAPI server receives the data:
    * Decodes the base64 audio and image.
    * The audio is saved temporarily and processed to extract melspectrograms.
    * A face is detected in the image using the `face-alignment` library.
4.  The **Wav2Lip** model processes the image and audio features:
    * It generates video frames where the lips in the input image are synced with the provided audio.
    * These frames are initially compiled into a video stream (e.g., AVI format).
5.  The generated video (without audio) is then combined with the original input audio using `ffmpeg` to create a final MP4 video.
6.  This final MP4 video is encoded into a base64 string.
7.  The server sends a JSON response back to the client over WebSocket, containing the base64 encoded video and a status message.
Models (Wav2Lip, face detector, and optional face segmentation/super-resolution) are loaded once at application startup for efficiency.

---

## Project Structure

It's assumed your project is structured as follows for the Docker build and local execution (especially `main.py` and other modules being inside an `app` directory):
```
.
├── app/
│   ├── main.py                 # FastAPI application
│   ├── processing.py           # Core lip-sync logic
│   ├── wav2lip_models.py       # Wav2Lip model definition
│   ├── face_detection.py       # Face detection utilities
│   ├── face_parsing.py         # Face segmentation utilities (optional)
│   ├── audio.py                # Audio processing utilities
│   ├── utilis.py               # General utilities (e.g., base64 conversion)
│   ├── config.py               # Configuration settings (e.g., model paths)
│   ├── schemas.py              # Pydantic schemas for input data
│   └── basicsr/                # Directory for BasicSR super-resolution (optional)
├── checkpoints/                # Directory for pre-trained model files
│   ├── wav2lip_gan.pth
│   └── ... (other model files like segmentation, sr if used)
├── input_image/
│   └── Obama.jpg               # Sample image for client testing
├── input_audios/
│   └── ai.wav                  # Sample audio for client testing
├── client.py                   # Example WebSocket client
├── Dockerfile
├── requirements.txt
└── README.md
```

## Prerequisites

* Python 3.10 (as per `Dockerfile`)
* `ffmpeg`: Essential for video and audio processing. Ensure it's installed and accessible in your system's PATH (or installed within Docker).
* Docker (for containerized deployment)
* For GPU support:
    * NVIDIA drivers on the host machine.
    * NVIDIA Container Toolkit installed on the host.

---

## Installation

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khoatran02/WebSocket_Lip-syncing.git
    cd WebSocket_Lip-syncing
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file should list all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    Ensure all pretrained modules (e.g., `wav2lip_models`, `face_detection`) are correctly placed within the `app/checkpoints` directory.

### Downloading Pretrained Models

The application requires pretrained models to function.
1.  **Wav2Lip Model**:
    * Download `wav2lip_gan.pth` (the main Wav2Lip model). You can typically find this in the original Wav2Lip repository or related resources.
    * Place it in the `checkpoints/` directory (or the path specified in your `app/config.py` for `settings.wav2lip_path`).
    Example: `checkpoints/wav2lip_gan.pth`

2.  **Face Detection Models**:
    * The `face-alignment` library (`face_detection.py` in your code) typically handles downloading its own pretrained models automatically upon first use or has them packaged. If it requires manual download for a specific version (like an `s3fd` model), refer to the `face-alignment` library's documentation.

3.  **(Optional) Face Segmentation Model**:
    * If you enable face segmentation (currently commented out in `processing.py`), download the required model (e.g., `79999_iter.pth` for BiSeNet).
    * Place it in the `checkpoints/` directory (or the path specified in `app/config.py` for `settings.segmentation_path`).

4.  **(Optional) Super Resolution Model**:
    * If you enable super-resolution (currently commented out in `processing.py`), download the required SR model.
    * Place it in the `checkpoints/` directory (or the path specified in `app/config.py` for `settings.sr_path`).

Ensure the paths in your configuration file (`app/config.py`, accessed via `settings.model_path`) correctly point to these downloaded files.

### Configuration

Model paths and other settings are likely managed in `app/config.py`. Ensure these paths are correct for your local setup or within the Docker container.

Example `app/config.py` structure (based on usage in `main.py`):
```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    wav2lip_path: str = "checkpoints/wav2lip_gan.pth"
    segmentation_path: str = "checkpoints/79999_iter.pth" # Example, if used
    sr_path: str = "checkpoints/your_sr_model.pth"      # Example, if used
    device: str = "cuda" # or "cpu"

settings = Settings()
```

### Downloading Pretrained Models

Provide clear instructions on where to download the necessary pretrained model files (e.g., `.pth` for PyTorch models) and where to place them.

Example for Wav2Lip:
1.  Download `wav2lip_gan.pth` from [Link to Wav2Lip GAN model].
2.  Download `s3fd_convert.pth` (for face detection) from [Link to S3FD model, if you use their detector].
3.  Place them in a designated directory, e.g., `checkpoints/`:
    ```
    checkpoints/
    ├── wav2lip_gan.pth
    └── s3fd_convert.pth # or other face detector model
    ```
---
## Running the Application

### Locally
1.  Ensure all dependencies and pretrained models are installed/downloaded and configured.
2.  Navigate to the directory containing the app/ folder.
3.  Run the FastAPI application using Uvicorn:
    ```
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
### Running with Docker

1.  **Build the Docker image:**
    From the root of the project directory (where your `Dockerfile` is):
    ```bash
    docker build -t lipsync-api .
    ```

2.  **Run the Docker container:**
    * **CPU:**
        ```bash
        docker run -d -p 8000:8000 --name lipsync-app lipsync-api
        ```
    * **GPU (if you configured Dockerfile for GPU and have NVIDIA Container Toolkit):**
        ```bash
        docker run -d --gpus all -p 8000:8000 --name lipsync-app lipsync-api
        ```
    The API will be accessible at `ws://localhost:8000/ws/lipsync`.

---

## API Usage

### WebSocket Endpoint
`ws://<your-server-address>:<port>/ws/lipsync`
(e.g., `ws://localhost:8000/ws/lipsync` when running locally or in Docker)

### Input Format
Send a JSON message with the following structure (as demonstrated in client.py):

```json
{
  "audio": "<base64_encoded_audio_bytes_wav_format>",
  "image": "<base64_encoded_image_bytes_jpg_or_png>",
  "fps": 25.0,
  "pads": [0, 10, 0, 0],
  "img_size": 96,
  "batch_size": 128,
  "no_segmentation": false,
  "no_sr": false
}
```
* audio: Base64 encoded string of the WAV audio file.
* image: Base64 encoded string of the person image (e.g., PNG, JPG). The image should ideally contain one clear frontal face.
* fps (float): Frames per second for the output video.
* pads (list of int): Padding around the detected face [top, bottom, left, right].
* img_size (int): The size to which the cropped face is resized before being fed into Wav2Lip (e.g., 96 for 96x96).
* batch_size (int): Batch size for Wav2Lip inference.
* no_segmentation (bool): Set to true to disable face segmentation (if the model is available and code enabled).
* no_sr (bool): Set to true to disable super-resolution (if the model is available and code enabled).

### Output Format
Successful Response:
```json
{
  "status": "success",
  "video": "<base64_encoded_video_bytes_mp4>"
}
```
* video: Base64 encoded string of the generated MP4 video.
Error Response:
```json
{
  "status": "success",
  "video": "<base64_encoded_video_bytes_mp4>"
}
```

### Testing with a WebSocket Client

You can use the provided client.py to test the API.

1.  Ensure the FastAPI server is running (either locally or in Docker).
2.  Place sample files:
* Put a sample image (e.g., Obama.jpg) in an input_image/ directory relative to where you run client.py.
* Put a sample audio file (e.g., ai.wav) in an input_audios/ directory relative to where you run client.py.
* Update the paths in client.py if your files are named or located differently.

3. Run the client script:
```
python client.py
```
The client will:
* Encode the sample image and audio to base64.
* Send the data to the WebSocket server.
* Receive the response.
* If successful, decode the base64 video and save it as output.mp4 in the same directory where client.py is run.
* Log messages about the process.







