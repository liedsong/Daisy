import re

class ChatParser:
    def __init__(self):
        # Regex for common timestamps (HH:MM, YYYY-MM-DD, Yesterday, etc.)
        self.time_patterns = [
            r'^\d{1,2}:\d{2}$',              # 12:30
            r'^\d{1,2}:\d{2}\s?[AP]M$',      # 12:30 PM
            r'^(Yesterday|Today|明天|昨天|今天)$',
            r'^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?$' # 2023-10-01
        ]

    def is_timestamp(self, text):
        clean_text = text.strip()
        for pattern in self.time_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return True
        return False

    def parse(self, ocr_results, image_width):
        """
        Parse OCR results into structured chat messages.
        Args:
            ocr_results: List of [box, text, conf] from EasyOCR.
            image_width: Width of the image.
        Returns:
            list of dict: Structured messages.
        """
        messages = []
        
        # Sort results by Y coordinate (top to bottom)
        # box[0] is top-left corner [x, y]
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])

        for bbox, text, conf in sorted_results:
            # Filter out low confidence
            if conf < 0.3:
                continue

            # Filter timestamps
            if self.is_timestamp(text):
                continue

            # Calculate center X
            x_coords = [p[0] for p in bbox]
            center_x = sum(x_coords) / 4
            
            # Determine role
            # 60% threshold for "Me" (usually right aligned)
            # 40% threshold for "Target" (usually left aligned)
            # Center (40-60%) might be system message or timestamp (already filtered)
            
            if center_x > image_width * 0.6:
                role = 'me'
            elif center_x < image_width * 0.4:
                role = 'target'
            else:
                # Ambiguous - treat as system or ignore? 
                # For now, if it's not a timestamp, maybe it's a short message.
                # Let's default to target if closer to left, me if closer to right.
                role = 'me' if center_x > image_width * 0.5 else 'target'

            messages.append({
                'role': role,
                'text': text,
                'confidence': conf
            })
            
        return messages
