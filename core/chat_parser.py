import re

class ChatParser:
    def parse(self, ocr_results, image_width):
        """
        Parses raw OCR results into structured chat messages based on X-coordinates.
        Filters out timestamps and non-message elements.
        
        Args:
            ocr_results: List of [box, text, confidence] from EasyOCR.
            image_width: Width of the image to determine left/right alignment.
            
        Returns:
            List of dict: [{'role': 'me'|'target', 'text': '...'}]
        """
        messages = []
        center_x = image_width / 2
        
        # Heuristic: 
        # Left side (< center) -> Target
        # Right side (> center) -> Me
        # Center aligned -> System message / Time stamp (Ignore)
        
        for result in ocr_results:
            box, text, conf = result
            
            # Filter low confidence
            if conf < 0.3:
                continue
                
            # Calculate center of the text box
            # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [p[0] for p in box]
            box_center_x = sum(x_coords) / 4
            
            # Filter timestamps
            # Common patterns: "12:30", "Yesterday 14:00", "2023年10月1日", "上午 10:00"
            # Regex for time/date patterns
            time_pattern = r'^\d{1,2}:\d{2}$|^\d{4}年|^\d{1,2}月\d{1,2}日|^Yesterday|^Today|^上午|^下午|^中午|^凌晨|^晚上'
            if re.search(time_pattern, text.strip()):
                continue
            
            # Determine role based on position
            # Add a buffer zone (e.g., 10% of width) for center alignment check
            buffer = image_width * 0.1
            
            if abs(box_center_x - center_x) < buffer:
                # Likely a center-aligned system message or timestamp -> Ignore
                continue
            elif box_center_x < center_x:
                role = 'target'
            else:
                role = 'me'
                
            # Merge with previous message if role is same and vertical distance is small?
            # For simplicity, we just append. Advanced logic can merge bubbles.
            messages.append({
                'role': role,
                'text': text
            })
            
        return messages
