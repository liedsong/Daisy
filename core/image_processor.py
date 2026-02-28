import cv2
import numpy as np
from typing import List, Optional
from core.logger import get_logger

logger = get_logger(__name__)

class ImageProcessor:
    """
    Handles image preprocessing including denoising and vertical stitching for chat screenshots.
    """

    @staticmethod
    def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Removes noise using fastNlMeansDenoisingColored.
        Effective for JPEG artifacts in screenshots.
        Args:
            image: Input image (BGR)
            strength: Denoising strength (higher = more blur)
        Returns:
            Denoised image
        """
        try:
            logger.info(f"Denoising image (strength={strength})...")
            # Parameters: h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
            return denoised
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return image

    @staticmethod
    def sharpen_image(image: np.ndarray) -> np.ndarray:
        """
        Applies a sharpening kernel to enhance text edges for OCR.
        Args:
            image: Input image
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def find_vertical_offset(img1: np.ndarray, img2: np.ndarray) -> Optional[int]:
        """
        Finds the vertical offset to stitch img2 below img1 using template matching.
        Assumes strict vertical overlap (chat history scrolling).
        Args:
            img1: Top image
            img2: Bottom image
        Returns:
            Vertical offset (y-coordinate in img1 where img2 starts), or None if no match found.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Ensure widths match by resizing img2 to img1's width
        if w1 != w2:
            new_h2 = int(h2 * w1 / w2)
            img2 = cv2.resize(img2, (w1, new_h2))
            h2, w2 = img2.shape[:2]

        # Use the top 20% of the second image as the template
        template_h = max(50, int(h2 * 0.2))
        template = img2[0:template_h, :]

        # Search in the bottom 50% of the first image for performance
        search_start_y = int(h1 * 0.5)
        search_region = img1[search_start_y:, :]
        
        try:
            res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Threshold for a "good" match
            if max_val > 0.8:
                # Calculate the y-coordinate in the original img1
                match_y_in_search = max_loc[1]
                match_y_global = search_start_y + match_y_in_search
                return match_y_global
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")

        return None

    @classmethod
    def stitch_vertical(cls, images: List[np.ndarray]) -> np.ndarray:
        """
        Stitches a list of images vertically, handling overlaps.
        Args:
            images: List of images (BGR numpy arrays)
        Returns:
            Single stitched image
        """
        if not images:
            raise ValueError("No images provided for stitching")
        
        if len(images) == 1:
            return images[0]

        # Start with the first image
        result = images[0]

        for i in range(1, len(images)):
            next_img = images[i]
            
            # Find overlap between current result (bottom part) and next_img (top part)
            # We only look at the bottom of result to save time
            offset_y = cls.find_vertical_offset(result, next_img)
            
            h_res, w_res = result.shape[:2]
            h_next, w_next = next_img.shape[:2]

            # Resize next_img to match result's width if needed
            if w_res != w_next:
                new_h = int(h_next * w_res / w_next)
                next_img = cv2.resize(next_img, (w_res, new_h))

            if offset_y is not None:
                logger.info(f"Stitching image {i}: Found overlap at y={offset_y}")
                # Crop result to the overlap point and append next_img
                # result = result[:offset_y, :]  <-- This cuts off the overlap from top, we want to append bottom
                # Actually, offset_y is where the *template* (top of next_img) matched in *result*
                # So result should be kept up to offset_y, and then append next_img
                result = np.vstack((result[:offset_y, :], next_img))
            else:
                logger.warning(f"Stitching image {i}: No overlap found, appending directly.")
                result = np.vstack((result, next_img))

        return result

    @classmethod
    def process(cls, images: List[np.ndarray], denoise: bool = False) -> np.ndarray:
        """
        Main pipeline: Denoise -> Stitch -> Sharpen (Optional)
        """
        if denoise:
            processed_images = [cls.denoise_image(img) for img in images]
        else:
            processed_images = images

        stitched_image = cls.stitch_vertical(processed_images)
        
        return stitched_image
