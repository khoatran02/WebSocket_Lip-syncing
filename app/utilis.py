import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""

    img_data = base64.b64decode(base64_string)
    pil_image = Image.open(BytesIO(img_data))
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencv_image

def image_to_base64(opencv_image):
    """Convert OpenCV image to base64 string"""
    
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_audio_from_bytes(audio_bytes, audio_save_path, original_sr):
    """
    Convert audio bytes back to a WAV file and save it.
    
    Parameters:
    - audio_bytes: Bytes object containing the audio data
    - audio_save_path: Path where the WAV file should be saved
    - original_sr: Original sample rate of the audio
    """
    # Convert bytes back to numpy array
    # We use np.frombuffer since the data was originally from tobytes()
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)  # or dtype=np.int16 for 16-bit PCM
    
    # Save the array as a WAV file
    sf.write(audio_save_path, audio_array, original_sr)
    
    print(f"Audio successfully saved to {audio_save_path}")


def audio_to_base64(wav_file_path):
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
        print(f"Error: The file '{wav_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while converting '{wav_file_path}' to base64: {e}")
        return None

def base64_to_audio(base64_encoded_string, output_wav_path):
    """
    Converts a base64 encoded string back into a .wav file and saves it.

    Parameters:
    base64_encoded_string (str): The base64 encoded string representing the WAV file.
    output_wav_path (str): The full path (including filename.wav) where the
                             reconstructed audio file will be saved.

    Returns:
    bool: True if the conversion and saving were successful, False otherwise.
    """
    try:
        # The base64.b64decode function expects bytes or an ASCII string.
        # If your string is already a standard Python string from .decode('utf-8'),
        # it should work directly as it contains only ASCII-compatible characters.
        # Alternatively, you could encode it to ASCII: base64_encoded_string.encode('ascii')
        print(f"Attempting to decode base64 string and save to: {output_wav_path}")
        
        wav_bytes = base64.b64decode(base64_encoded_string)
        
        # Write the decoded bytes to a new WAV file in binary write mode ('wb')
        with open(output_wav_path, 'wb') as wav_file:
            wav_file.write(wav_bytes)
            
        print(f"✅ Successfully converted base64 string to WAV file: {output_wav_path}")
        return True
        
    except binascii.Error as e:
        # This error is raised if the input string is not valid base64
        print(f"Error: Invalid base64 string provided. Details: {e}")
        return False
    except FileNotFoundError:
        # This might occur if the output_wav_path is invalid (e.g., directory doesn't exist)
        print(f"Error: Could not write to path '{output_wav_path}'. Please ensure the directory exists and path is valid.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while converting base64 to WAV: {e}")
        return False


import base64

def video_to_base64(video_path):
    """
    Converts a video file to its base64 encoded string representation.

    Parameters:
    video_path (str): The full path to the video file (e.g., "my_video.mp4").

    Returns:
    str: The base64 encoded string of the video file.
         Returns None if the file is not found or an error occurs during conversion.
    """
    try:
        print(f"Attempting to open video file: {video_path}")
        # Open the video file in binary read mode ('rb')
        with open(video_path, 'rb') as video_file:
            # Read the entire binary content of the file
            video_bytes = video_file.read()
            print(f"Successfully read {len(video_bytes)} bytes from the video file.")

            # Encode the binary content to base64
            base64_encoded_bytes = base64.b64encode(video_bytes)
            print("Video bytes successfully encoded to base64.")

            # Decode the base64 bytes to a UTF-8 string
            base64_encoded_string = base64_encoded_bytes.decode('utf-8')
            print("Base64 bytes successfully decoded to UTF-8 string.")
            
        return base64_encoded_string
        
    except FileNotFoundError:
        print(f"Error: The video file '{video_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while converting '{video_path}' to base64: {e}")
        return None