from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Detection:
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime

@dataclass
class Visit:
    detection: Detection
    thumbnail_path: str
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None