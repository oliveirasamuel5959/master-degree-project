from pathlib import Path
from typing import List

class DataLoader:
  def __init__(self, path: Path):
    self.data_path = path

  def get_video_files(self) -> List[Path]:
    video_files = list(self.data_path.rglob("*.mp4"))
    return video_files