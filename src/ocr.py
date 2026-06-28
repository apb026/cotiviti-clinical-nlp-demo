"""
ocr.py
Optical Character Recognition using free, open-source Tesseract OCR
(via pytesseract). Runs fully locally - no cloud OCR API required.

Requires the Tesseract OCR binary to be installed on the system:
  - Ubuntu/Debian: sudo apt-get install tesseract-ocr
  - macOS:         brew install tesseract
  - Windows:       https://github.com/UB-Mannheim/tesseract/wiki
"""

from PIL import Image
import pytesseract


def extract_text_from_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL Image and return extracted text."""
    text = pytesseract.image_to_string(image)
    return text.strip()


def check_tesseract_installed() -> bool:
    """Return True if the Tesseract binary is available on this system."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
