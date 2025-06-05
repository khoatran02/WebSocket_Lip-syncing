from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    audio: str  # base64 encoded audio
    image: str  # base64 encoded image
    fps: float = 25.0
    pads: List[int] = [0, 10, 0, 0]
    img_size: int = 96
    batch_size: int = 128
    no_segmentation: bool = False
    no_sr: bool = False

class OutputData(BaseModel):
    status: str  # "success" or "error"
    video: Optional[str] = None  # base64 encoded video
    message: Optional[str] = None  # error message