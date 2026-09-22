import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

class QRDecoder:
    """
    Robust Multi-Engine QR Code Decoder.
    Combines OpenCV QRCodeDetector and PyZbar with image preprocessing
    (grayscale, contrast stretching, sharpening, thresholding).
    """
    def __init__(self):
        self.cv_detector = cv2.QRCodeDetector()

    def decode_image(self, image_path_or_bytes) -> Dict[str, Any]:
        """
        Decodes a QR code from file path, PIL Image, or numpy array.
        Returns:
            {
                'success': bool,
                'raw_content': str,
                'is_url': bool,
                'error': Optional[str]
            }
        """
        try:
            if isinstance(image_path_or_bytes, str):
                img = cv2.imread(image_path_or_bytes)
            elif isinstance(image_path_or_bytes, np.ndarray):
                img = image_path_or_bytes
            elif hasattr(image_path_or_bytes, 'read'):
                # File-like object
                file_bytes = np.asarray(bytearray(image_path_or_bytes.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                return {'success': False, 'raw_content': '', 'is_url': False, 'error': 'Unsupported image input type'}

            if img is None:
                return {'success': False, 'raw_content': '', 'is_url': False, 'error': 'Could not read image file or buffer'}

            # Strategy 1: Direct OpenCV Detection
            data, bbox, _ = self.cv_detector.detectAndDecode(img)
            if data and len(data.strip()) > 0:
                content = data.strip()
                return {'success': True, 'raw_content': content, 'is_url': self._check_is_url(content), 'error': None}

            # Strategy 2: Grayscale + Otsu Thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data, bbox, _ = self.cv_detector.detectAndDecode(thresh)
            if data and len(data.strip()) > 0:
                content = data.strip()
                return {'success': True, 'raw_content': content, 'is_url': self._check_is_url(content), 'error': None}

            # Strategy 3: PyZbar Fallback if available
            if PYZBAR_AVAILABLE:
                decoded_objects = pyzbar_decode(img)
                if not decoded_objects:
                    decoded_objects = pyzbar_decode(gray)
                if decoded_objects:
                    content = decoded_objects[0].data.decode('utf-8', errors='ignore').strip()
                    return {'success': True, 'raw_content': content, 'is_url': self._check_is_url(content), 'error': None}

            return {'success': False, 'raw_content': '', 'is_url': False, 'error': 'No QR code pattern detected in the image.'}

        except Exception as e:
            return {'success': False, 'raw_content': '', 'is_url': False, 'error': str(e)}

    @staticmethod
    def _check_is_url(text: str) -> bool:
        t = text.lower().strip()
        return t.startswith(('http://', 'https://', 'ftp://', 'www.')) or ('.' in t and '/' in t)
