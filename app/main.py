from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from processing import load_models, process_audio_image
from schemas import InputData
from config import settings
from contextlib import asynccontextmanager
import json

# Load models at startup
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    models.update(load_models(
        wav2lip_path=settings.wav2lip_path,
        segmentation_path=settings.segmentation_path,
        super_resolution_path=settings.sr_path,
        device=settings.device
    ))
    yield
    # Clean up the ML models and release the resources
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/lip-sync")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:

            print("Client connected")
            # Receive data from client
            data = await websocket.receive_json()
            input_data = InputData(**data)
            
            # Process audio and image
            output_video = process_audio_image(
                models=models,
                input_data = input_data,
                fps=input_data.fps,
                pads=input_data.pads,
                img_size=input_data.img_size,
                wav2lip_batch_size=input_data.batch_size,
                segmentation=input_data.segmentation,
                super_resolution=input_data.super_resolution,
            )

            print("Video processing completed")
            
            # Send result back
            await websocket.send_json({
                "status": "success",
                "video": output_video
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })

@app.get("/")
async def get():
    return HTMLResponse("""
    <html>
        <head>
            <title>Wav2Lip WebSocket</title>
        </head>
        <body>
            <h1>WebSocket endpoint for real-time lip-syncing</h1>
            <p>Connect to ws://localhost:8000/ws/lip-sync</p>
        </body>
    </html>
    """)