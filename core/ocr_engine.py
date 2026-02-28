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
        # Add contrast_ths and adjust_contrast to improve accuracy on dark/low-contrast images
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

        # Preprocessing to improve accuracy
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Rescale if image is too small (width < 1000px)
        # EasyOCR works better on larger text
        h, w = gray.shape
        if w < 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            logger.info(f"Upscaled image for OCR by factor {scale:.2f}")

        # Perform OCR
        # detail=1 returns bounding box, text, and confidence
        # paragraph=False ensuring line-by-line detection
        results = self.reader.readtext(gray, detail=1, paragraph=False)
        return results

if __name__ == "__main__":
    # Simple test
    engine = OCREngine()
    print("OCR Engine initialized successfully.")
