import asyncio
import websockets
import json
import base64
import cv2
import soundfile as sf
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wav_to_base64(wav_file_path):
    """
    Converts a WAV audio file to a base64 encoded string.

    This function reads the entire WAV file (including headers and data)
    and encodes it.

    Parameters:
    wav_file_path (str): The path to the .wav file.

    Returns:
    str: The base64 encoded string representation of the WAV file.
         Returns None if the file is not found or an error occurs.
    """
    try:
        # Open the WAV file in binary read mode ('rb')
        with open(wav_file_path, 'rb') as wav_file:
            # Read the entire binary content of the file
            wav_bytes = wav_file.read()
            
            # Encode the binary content to base64
            base64_encoded_bytes = base64.b64encode(wav_bytes)
            
            # Decode the base64 bytes to a UTF-8 string
            base64_encoded = base64_encoded_bytes.decode('utf-8')
            
        return base64_encoded
        
    except FileNotFoundError:
        logger.error(f"Error: The file '{wav_file_path}' was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while converting '{wav_file_path}' to base64: {e}")
        return None

async def send_audio_image():
    uri = "ws://localhost:8000/ws/lip-sync"
    
    try:
        # Create a connection with increased timeouts
        async with websockets.connect(
            uri,
            ping_timeout=300,  # 5 minutes
            close_timeout=300,  # 5 minutes
            max_size=None       # Remove size limit for large files
        ) as websocket:
            
            logger.info("Connected to WebSocket server")
            
            # Load sample image and audio
            image = cv2.imread("input_image/Obama.jpg")
            if image is None:
                raise ValueError("Failed to load image")
                
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info("Image encoded to base64")

            audio_base64 = wav_to_base64("input_audios/ai.wav")
            if not audio_base64:
                raise ValueError("Failed to convert audio to base64")
            logger.info("Audio encoded to base64")

            # Prepare the request data
            request_data = {
                "audio": audio_base64,
                "image": image_base64,
                "fps": 25.0,
                "pads": [0, 10, 0, 0],
                "img_size": 96,
                "batch_size": 128,
                "segmentation": False,
                "super_resolution": False,
            }
            
            # Send data with timeout
            logger.info("Sending data to server...")
            await asyncio.wait_for(
                websocket.send(json.dumps(request_data)),
                timeout=30.0  # 30 seconds to send
            )
            logger.info("Data sent successfully")
            
            # Receive response with extended timeout
            logger.info("Waiting for server response...")
            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=600.0  # 10 minutes for processing
            )
            logger.info("Received response from server")
            
            result = json.loads(response)
            # print("Response from server:", result)
            
            # Save video if successful
            if result.get("status") == "success":
                video_data = base64.b64decode(result["video"])
                with open("output.mp4", "wb") as f:
                    f.write(video_data)
                logger.info("Video saved as output.mp4")
            else:
                logger.error(f"Server returned error: {result.get('message', 'Unknown error')}")
                
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e.code} - {e.reason}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

async def main():
    await send_audio_image()

if __name__ == "__main__":
    asyncio.run(main())