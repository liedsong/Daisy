import easyocr
import cv2
import numpy as np
from core.logger import get_logger

logger = get_logger(__name__)

class OCREngine:
    def __init__(self, languages=['ch_sim', 'en'], use_gpu=False):
        """
        Initialize the EasyOCR reader.
        Args:
            languages (list): List of languages to support. Default is Simplified Chinese and English.
            use_gpu (bool): Whether to use GPU for inference.
        """
        # Force CPU if CUDA is not available
        import torch
        if use_gpu and not torch.cuda.is_available():
            logger.warning("CUDA is not available. Falling back to CPU.")
            use_gpu = False

        logger.info(f"Initializing EasyOCR with GPU={use_gpu}...")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def extract_text(self, image_path_or_bytes):
        """
        Extract text from an image.
        Args:
            image_path_or_bytes: File path or bytes of the image.
        Returns:
            list: Raw result from EasyOCR [[box, text, confidence], ...]
        """
        # If input is bytes (from Streamlit upload), convert to numpy array for OpenCV
        if isinstance(image_path_or_bytes, bytes):
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_path_or_bytes

        # Perform OCR
        # detail=1 returns bounding box, text, and confidence
        results = self.reader.readtext(image, detail=1)
        return results

if __name__ == "__main__":
    # Simple test
    engine = OCREngine()
    print("OCR Engine initialized successfully.")
