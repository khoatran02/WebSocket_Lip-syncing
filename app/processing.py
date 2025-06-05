import torch
import numpy as np
import cv2, os, subprocess
from wav2lip_models import Wav2Lip
from face_parsing import init_parser, swap_regions
from basicsr.apply_sr import init_sr_model, enhance
import face_detection
import audio
import av
import io
import soundfile as sf
from utilis import base64_to_image, base64_to_audio, video_to_base64
import tempfile


mel_step_size = 16

def load_models(wav2lip_path, segmentation_path, super_resolution_path, device='cuda'):
    """Load all required models"""
    models = {}
    
    # Load Wav2Lip model
    print("Loading Wav2Lip model...")
    wav2lip = Wav2Lip()
    checkpoint = torch.load(wav2lip_path, map_location=device)
    s = checkpoint["state_dict"]
    wav2lip.load_state_dict({k.replace('module.', ''): v for k, v in s.items()})
    models['wav2lip'] = wav2lip.to(device).eval()
    
    #Load face segmentation model if path provided
    # if segmentation_path:
    #     print("Loading face segmentation model...")
    #     models['seg_net'] = init_parser(segmentation_path)
    # else:
    #     models['seg_net'] = None
    
    # # Load super-resolution model if path provided
    # if super_resolution_path:
    #     print("Loading super-resolution model...")
    #     models['sr_net'] = init_sr_model(super_resolution_path)
    # else:
    #     models['sr_net'] = None
    
    # Initialize face detector
    models['detector'] = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device=device
    )
    
    return models

def face_detect(image, detector, pads=[0, 10, 0, 0]):
    """Detect face in image with padding"""
    predictions = detector.get_detections_for_batch(np.array([image]))
    
    pady1, pady2, padx1, padx2 = pads
    rect = predictions[0]
    if rect is None:
        raise ValueError('Face not detected! Try different padding values.')
    
    y1 = max(0, rect[1] - pady1)
    y2 = min(image.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(image.shape[1], rect[2] + padx2)
    
    return image[y1:y2, x1:x2], (y1, y2, x1, x2)

def combine_audio_video_in_memory(audio_bytes, video_path, output_path):
    """
    Combine audio and video without saving intermediate audio file
    using FFmpeg's pipe protocol
    """
    # FFmpeg command that reads audio from stdin
    ffmpeg_cmd = [
        'C:/ffmpeg/bin/ffmpeg.exe',
        '-y',  # overwrite output
        '-f', 'wav',  # input format
        '-i', 'pipe:0',  # read audio from stdin
        '-i', video_path,  # video input
        '-c:v', 'copy',  # copy video stream
        '-c:a', 'aac',  # encode audio as AAC
        '-strict', '-2',  # allow experimental codecs
        output_path
    ]
    
    # Run FFmpeg with audio bytes piped to stdin
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate(input=audio_bytes)
    
    if process.returncode != 0:
        raise RuntimeError("FFmpeg failed to combine audio and video")

def datagen(models, mels, image, pads, img_size, wav2lip_batch_size):
    img_batch, mel_batch, coords_batch = [], [], []
    face_crop, coords = face_detect(image, models['detector'], pads)
    
    for m in mels:
        face = cv2.resize(face_crop, (img_size, img_size))
        img_masked = face.copy()
        img_masked[:, img_size//2:] = 0
        
        img_batch.append(face)
        mel_batch.append(m)
        coords_batch.append(coords)
        
        if len(img_batch) >= wav2lip_batch_size:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)
            
            img_masked_np = img_batch_np.copy()
            img_masked_np[:, img_size//2:] = 0
            
            img_batch_np = np.concatenate((img_masked_np, img_batch_np), axis=3) / 255.
            mel_batch_np = np.reshape(mel_batch_np, 
                                     [len(mel_batch_np), mel_batch_np.shape[1], 
                                      mel_batch_np.shape[2], 1])
            
            yield img_batch_np, mel_batch_np, coords_batch
            img_batch, mel_batch, coords_batch = [], [], []
    
    if img_batch:
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        
        img_masked_np = img_batch_np.copy()
        img_masked_np[:, img_size//2:] = 0
        
        img_batch_np = np.concatenate((img_masked_np, img_batch_np), axis=3) / 255.
        mel_batch_np = np.reshape(mel_batch_np, 
                                 [len(mel_batch_np), mel_batch_np.shape[1], 
                                  mel_batch_np.shape[2], 1])
        
        yield img_batch_np, mel_batch_np, coords_batch

def process_audio_image(models, input_data, fps=25., pads=[0, 10, 0, 0], 
                       img_size=96, wav2lip_batch_size=128, segmentation=False, super_resolution=False):
    
    """Process audio and image to generate lip-synced video"""
    device = next(models['wav2lip'].parameters()).device

    with tempfile.TemporaryDirectory() as temp_dir:

        # Decode base64 data
        image = base64_to_image(input_data.image)   
        if image is None:
            raise ValueError("Invalid image data")    

        # Convert audio from base64
        audio_path = os.path.join(temp_dir, "audio.wav")
        success = base64_to_audio(input_data.audio, audio_path)
        if not success:
            raise ValueError("Failed to decode audio from base64")

        # Process audio
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Add small noise to the audio and try again')
        
        # Prepare mel chunks
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        
        # Process image
        frame_h, frame_w = image.shape[:-1]
        temp_video_path = os.path.join(temp_dir, "result.avi")
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        # Generate frames
        for img_batch, mel_batch, coords in datagen(models, mel_chunks, image, pads, img_size, wav2lip_batch_size):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            
            with torch.no_grad():
                pred = models["wav2lip"](mel_batch, img_batch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, (y1, y2, x1, x2) in zip(pred, coords):
                # if super_resolution and sr_net:
                #     p = enhance(sr_net, p)
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                
                # if segmentation and seg_net:
                #     p = swap_regions(face_image[y1:y2, x1:x2], p, seg_net)
                
                output_frame = image.copy()
                output_frame[y1:y2, x1:x2] = p
                out.write(output_frame)
            
        out.release()
        
        video_path = os.path.join(temp_dir, "output.mp4")
        # Combine video and audio
        # ffmpeg_path = 'C:/ffmpeg/bin/ffmpeg.exe'  # Change if ffmpeg is not in PATH
        # command = f'{ffmpeg_path} -y -i {audio_path} -i {temp_video_path} -strict -2 -q:v 1 {video_path}'
                # Combine video and audio
        ffmpeg_path = 'ffmpeg'  
        command = [
            ffmpeg_path, '-y',
            '-i', audio_path,
            '-i', temp_video_path,
            '-strict', '-2',
            '-q:v', '1',
            video_path
        ]

        # subprocess.call(command, shell=True)
        subprocess.run(command, check=True)

           
        # Convert output video to base64
        video_base64 = video_to_base64(video_path)
    
    return video_base64

