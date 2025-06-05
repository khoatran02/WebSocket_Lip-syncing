from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    wav2lip_path: str = "checkpoints/wav2lip_gan.pth"
    segmentation_path: str = "checkpoints/face_segmentation.pth"
    sr_path: str = "checkpoints/esrgan_yunying.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    class Config:
        env_file = ".env"

settings = Settings()

