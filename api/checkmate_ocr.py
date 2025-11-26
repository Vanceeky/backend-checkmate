# checkmate_ocr.py

import io
import re
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
from google.cloud import vision
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# ---------------------------
# 1. Preprocessing Functions
# ---------------------------

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to OpenCV format, grayscale, threshold, remove underlines.
    """
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to highlight handwriting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Remove horizontal lines (underlines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)  # fill with black (remove line)

    # Invert back to normal text
    processed = cv2.bitwise_not(thresh)
    return processed

# ---------------------------
# 2. Google Vision OCR
# ---------------------------

def run_google_vision(cv_image: np.ndarray):
    """
    Send image to Google Vision and return OCR response.
    """
    client = vision.ImageAnnotatorClient()
    
    # Convert OpenCV image to bytes
    is_success, buffer = cv2.imencode(".png", cv_image)
    image_bytes = io.BytesIO(buffer).getvalue()
    image = vision.Image(content=image_bytes)
    
    response = client.document_text_detection(image=image)
    return response

# ---------------------------
# 3. Helper Functions
# ---------------------------

def extract_text_blocks(vision_response):
    """
    Extract text blocks with bounding boxes.
    Returns list of dicts: {text, bbox}
    """
    blocks = []
    for page in vision_response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = ''
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    block_text += word_text + ' '
            block_text = block_text.strip()
            # Get bounding box (xmin, ymin, xmax, ymax)
            vertices = block.bounding_box.vertices
            bbox = [vertices[0].x, vertices[0].y, vertices[2].x, vertices[2].y]
            blocks.append({"text": block_text, "bbox": bbox})
    return blocks

""" def find_student_id(blocks):

    Finds the student ID by locating "Student ID:" printed text
    and extracting handwriting to the right.
    
    student_id = ""
    for block in blocks:
        if "Student ID" in block["text"]:
            # For simplicity, assume handwriting is after colon on same line
            # Try to extract anything after ':' or just take the block next to it
            match = re.search(r"Student ID[:\s]*(\S+)", block["text"], re.IGNORECASE)
            if match:
                student_id = match.group(1)
            break
    return student_id """

def find_student_id(blocks):
    """
    Finds the student ID by locating "Student ID:" printed text
    and extracting handwriting to the right.
    """
    student_id = ""
    for block in blocks:
        # Check if "Student ID" is present (case-insensitive for robustness)
        if re.search(r"Student ID", block["text"], re.IGNORECASE):
            # Regex explanation:
            # r"Student ID[:\s-]*" : Matches "Student ID", followed by an optional colon, 
            #                        whitespace, or hyphen.
            # r"([\w-]+)"          : CAPTURES one or more (the '+') of word characters (\w) 
            #                        or hyphens (-). This specifically targets the ID format 
            #                        "18-9398-54S" and stops before the next word or newline.
            match = re.search(r"Student ID[:\s-]*([\w-]+)", block["text"], re.IGNORECASE)
            
            if match:
                student_id = match.group(1)
                break
    return student_id

def detect_sections(blocks):
    """
    Detect all sections by looking for "Test X: SECTION_NAME"
    Returns list of dicts: {"test_no": int, "section": str}
    """
    sections = []
    for block in blocks:
        match = re.match(r'Test\s*(\d+)\s*:\s*(.+)', block["text"], re.IGNORECASE)
        if match:
            test_no = int(match.group(1))
            section_name = match.group(2).strip().upper()
            sections.append({"test_no": test_no, "section": section_name})
    return sections

def extract_questions(blocks, section_name):
    """
    Extract questions for Identification & True/False.
    Returns list of dicts: {"q_no": int, "answer": str}
    """
    questions = []
    # Find all question numbers (e.g., 1., 2., 3.)
    for block in blocks:
        q_match = re.match(r'(\d+)\.', block["text"])
        if q_match:
            q_no = int(q_match.group(1))
            # Handwriting assumed before the number
            # Very simplified: remove underscores & take text before number
            answer_text = block["text"].split(str(q_no)+'.')[0].strip()
            answer_text = re.sub(r'[_]+', '', answer_text).strip()
            # Normalize True/False if needed
            if section_name == "TRUE OR FALSE":
                answer_text = answer_text.upper()
                if answer_text not in ["TRUE", "FALSE"]:
                    answer_text = ""  # mark as empty if not valid
            questions.append({"q_no": q_no, "answer": answer_text})
    return questions

def placeholder_questions():
    """Return empty questions for unimplemented sections"""
    return []

# ---------------------------
# 4. Main Processing Function
# ---------------------------

def process_exam_pdf(pdf_bytes):
    """
    Main function to process a PDF exam sheet.
    Returns JSON as required.
    """
    # Convert PDF to images
    pages = convert_from_bytes(pdf_bytes)
    
    all_blocks = []
    sections_data = []
    current_section_name = ""
    
    for page in pages:
        preprocessed_img = preprocess_image(page)
        vision_resp = run_google_vision(preprocessed_img)
        blocks = extract_text_blocks(vision_resp)
        all_blocks.extend(blocks)
        
        # Detect sections on this page
        page_sections = detect_sections(blocks)
        for sec in page_sections:
            current_section_name = sec["section"]
            # Trigger proper function
            if current_section_name in ["IDENTIFICATION", "TRUE OR FALSE"]:
                questions = extract_questions(blocks, current_section_name)
            elif current_section_name in ["ENUMERATION", "MULTIPLE CHOICE"]:
                questions = placeholder_questions()
            else:
                questions = placeholder_questions()
            
            sections_data.append({
                "section": current_section_name,
                "questions": questions
            })
    
    student_id = find_student_id(all_blocks)
    
    result = {
        "student_id": student_id,
        "sections": sections_data
    }
    
    return result
