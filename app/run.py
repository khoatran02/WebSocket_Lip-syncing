from os import path
import numpy as np
import cv2, os, subprocess
import torch, face_detection
import audio
from wav2lip_models import Wav2Lip
from face_parsing import init_parser, swap_regions
from basicsr.apply_sr import init_sr_model, enhance
import argparse

parser = argparse.ArgumentParser(description='Image to lip-sync video using Wav2Lip')

parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Path to saved Wav2Lip checkpoint')
parser.add_argument('--segmentation_path', type=str, required=True,
                    help='Path to face segmentation network checkpoint')
parser.add_argument('--sr_path', type=str, required=True,
                    help='Path to super-resolution network checkpoint')
parser.add_argument('--face', type=str, required=True,
                    help='Path to input image')
parser.add_argument('--audio', type=str, required=True,
                    help='Path to audio file')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4',
                    help='Output video path')
parser.add_argument('--fps', type=float, default=25.,
                    help='Frames per second for output video')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding for face detection (top, bottom, left, right)')
parser.add_argument('--img_size', type=int, default=96,
                    help='Size of face image for model input')
parser.add_argument('--wav2lip_batch_size', type=int, default=128,
                    help='Batch size for Wav2Lip model')
parser.add_argument('--no_segmentation', default=False, action='store_true',
					help='Prevent using face segmentation')
parser.add_argument('--no_sr', default=False, action='store_true',
					help='Prevent using super resolution')

args = parser.parse_args()
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    model = Wav2Lip()
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    model.load_state_dict({k.replace('module.', ''): v for k, v in s.items()})
    return model.to(device).eval()

def face_detect(image):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                          flip_input=False, device=device)
    predictions = detector.get_detections_for_batch(np.array([image]))
    
    pady1, pady2, padx1, padx2 = args.pads
    rect = predictions[0]
    if rect is None:
        raise ValueError('Face not detected! Try different padding values.')
    
    y1 = max(0, rect[1] - pady1)
    y2 = min(image.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(image.shape[1], rect[2] + padx2)
    
    return image[y1:y2, x1:x2], (y1, y2, x1, x2)

def datagen(mels, face_image):
    img_batch, mel_batch, coords_batch = [], [], []
    face_crop, coords = face_detect(face_image)
    
    for m in mels:
        face = cv2.resize(face_crop, (args.img_size, args.img_size))
        img_masked = face.copy()
        img_masked[:, args.img_size//2:] = 0
        
        img_batch.append(face)
        mel_batch.append(m)
        coords_batch.append(coords)
        
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)
            
            img_masked_np = img_batch_np.copy()
            img_masked_np[:, args.img_size//2:] = 0
            
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
        img_masked_np[:, args.img_size//2:] = 0
        
        img_batch_np = np.concatenate((img_masked_np, img_batch_np), axis=3) / 255.
        mel_batch_np = np.reshape(mel_batch_np, 
                                 [len(mel_batch_np), mel_batch_np.shape[1], 
                                  mel_batch_np.shape[2], 1])
        
        yield img_batch_np, mel_batch_np, coords_batch

def main():
    if not path.isfile(args.face):
        raise ValueError('--face must be a valid path to an image file')
    
    # Process audio
    if not args.audio.endswith('.wav'):
        print('Converting audio to WAV...')
        subprocess.call(f'ffmpeg -y -i {args.audio} -strict -2 temp/temp.wav', shell=True)
        args.audio = 'temp/temp.wav'
    
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Add small noise to the audio and try again')
    
    # Prepare mel chunks
    mel_chunks = []
    mel_idx_multiplier = 80./args.fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    # Load models
    print("Loading models...")
    # seg_net = init_parser(args.segmentation_path) if not args.no_segmentation else None
    # sr_net = init_sr_model(args.sr_path) if not args.no_sr else None
    model = load_model(args.checkpoint_path)
    
    # Process image
    face_image = cv2.imread(args.face)
    frame_h, frame_w = face_image.shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', 
                         cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))
    
    # Generate frames
    for img_batch, mel_batch, coords in datagen(mel_chunks, face_image):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, (y1, y2, x1, x2) in zip(pred, coords):
            # if not args.no_sr and sr_net:
            #     p = enhance(sr_net, p)
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # if not args.no_segmentation and seg_net:
            #     p = swap_regions(face_image[y1:y2, x1:x2], p, seg_net)
            
            output_frame = face_image.copy()
            output_frame[y1:y2, x1:x2] = p
            out.write(output_frame)
    
    out.release()
    
    # Combine video and audio
    ffmpeg_path = 'C:/ffmpeg/bin/ffmpeg.exe'  # Change if ffmpeg is not in PATH
    command = f'{ffmpeg_path} -y -i {args.audio} -i temp/result.avi -strict -2 -q:v 1 {args.outfile}'
    subprocess.call(command, shell=True)
    print(f"Output video saved to {args.outfile}")

if __name__ == '__main__':
    main()

# python run.py --checkpoint_path "checkpoints/wav2lip_gan.pth" --segmentation_path "checkpoints/face_segmentation.pth" --sr_path "checkpoints/esrgan_yunying.pth" --face "input_image/mona_lisa.png" --audio "input_audios/ai.wav" --outfile output/mona_lisa.mp4      
