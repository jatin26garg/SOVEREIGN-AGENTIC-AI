from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def ocr_image(image:  Image.Image, lang:str = "eng")->tuple[str,float]:
    
    data = pytesseract.image_to_data(
        image, lang=lang,output_type=pytesseract.Output.DICT
    )
    
    words,confs = [], []
    
    for word , conf in zip(data["text"], data["conf"])