import cv2
import torch, easyocr, numpy as np
from typing import Tuple, Optional

class LicensePlateReader:
    def __init__(self):
        """Initialize EasyOCR for license plate reading"""
        try:
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print(f"EasyOCR initialized with GPU: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            self.reader = None
    
    def read_plate(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[float]]:
        """Extract license plate text from image region"""
        if self.reader is None:
            return None, None
        
        try:
            x1, y1, x2, y2 = bbox
            # Expand bbox slightly for better OCR
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)
            
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return None, None
            
            # Enhance image for better OCR
            plate_region = cv2.resize(plate_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            plate_region = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            results = self.reader.readtext(plate_region)
            
            if results:
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                # Basic filtering for license plate patterns
                if len(text) >= 4 and confidence > 0.5:
                    return text.upper(), confidence
            
            return None, None
        
        except Exception as e:
            print(f"Error reading license plate: {e}")
            return None, None