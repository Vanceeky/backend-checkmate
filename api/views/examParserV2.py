import io
import os
import re
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from google.cloud import vision
from google.oauth2 import service_account
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Configuration / Helper flags
# -----------------------------
# You may set SERVICE_ACCOUNT_FILE env var to absolute path of your JSON key.
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
# If you set SERVICE_ACCOUNT_FILE to None, the client will attempt the default chain.
# Set DEBUG_SAVE_IMAGES = True to write debug images (with bubble centers / overlays) to /tmp
DEBUG_SAVE_IMAGES = True

# -----------------------------
# Init Vision client (robust)
# -----------------------------
def create_vision_client():
    if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        return vision.ImageAnnotatorClient(credentials=creds)
    else:
        # rely on Application Default Credentials if available
        return vision.ImageAnnotatorClient()

client = create_vision_client()


class ExamParser:
    def __init__(self, file_bytes, debug=False):
        self.file_bytes = file_bytes
        self.full_text = ""
        self.pages_images = []  # list of OpenCV BGR images
        self.student_id = ""
        self.parsed_sections = []
        self.debug = debug

        # --- Template OMR config: MUST be calibrated per your template (pixel coordinates on page image)
        # Provided are placeholders based on your template. You MUST update coordinates after running
        # debug overlay to match your scanned resolution (DPI).
        #
        # Structure:
        # MULTIPLE_CHOICE_TEMPLATE = {
        #   page_index: {
        #       question_number: { "A": (x,y,r), "B": (x,y,r), ... },
        #       ...
        #   }
        # }
        #
        # The (x,y) are pixel centers on the cv2 image and r is radius to sample.
        self.MULTIPLE_CHOICE_TEMPLATE = {
            # Example: assume MC on page 0
            0: {
                1: {"A": (150, 1200, 18), "B": (260, 1200, 18), "C": (370, 1200, 18)},
                2: {"A": (150, 1250, 18), "B": (260, 1250, 18), "C": (370, 1250, 18)}
                # Add more question entries as you calibrate
            }
        }

    # -------------------------------
    # Vision / OCR Helpers
    # -------------------------------
    def _get_vision_text(self, pil_image):
        """Sends image to Google Vision API to get text."""
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        return response.full_text_annotation.text

    def _pil_to_cv2(self, pil_image):
        """Convert PIL to cv2 BGR"""
        open_cv_image = np.array(pil_image)
        return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # -------------------------------
    # Core extraction
    # -------------------------------
    def extract_data(self):
        pil_images = convert_from_bytes(self.file_bytes)
        for i, pil_img in enumerate(pil_images):
            try:
                self.pages_images.append(self._pil_to_cv2(pil_img))
                text = self._get_vision_text(pil_img)
                self.full_text += f"\n{text}\n"
            except Exception as e:
                # do not crash; continue processing pages
                print(f"[extract_data] page {i} vision error: {e}")
                # still append image for OMR even if OCR failed
                if i >= len(self.pages_images):
                    self.pages_images.append(self._pil_to_cv2(pil_img))

        # Extract Student ID robustly (allow letters, digits, hyphens, spaces)
        sid_match = re.search(r"Student\s*ID\s*[:\-]?\s*([A-Za-z0-9\-\s]+)", self.full_text, re.IGNORECASE)
        if sid_match:
            self.student_id = sid_match.group(1).strip()

        # Split into test blocks using a forgiving regex (handles OCR noise/newlines)
        pattern = r"(Test\s*\d+\s*[:.\-]?\s*[A-Za-z ]+)([\s\S]*?)(?=Test\s*\d+|$)"
        blocks = re.findall(pattern, self.full_text, re.IGNORECASE)

        for header, body in blocks:
            header_up = header.upper()
            if "IDENTIFICATION" in header_up:
                questions = self.parse_pre_answer(body)
                section_type = "IDENTIFICATION"
            elif "TRUE" in header_up and "FALSE" in header_up:
                questions = self.parse_pre_answer(body)
                section_type = "TRUE_OR_FALSE"
            elif "ENUMERATION" in header_up:
                questions = self.parse_enumeration(body)
                section_type = "ENUMERATION"
            elif "MULTIPLE" in header_up and "CHOICE" in header_up:
                # Use template-based OMR first; fallback to text detection
                questions = self.parse_omr_with_template()
                section_type = "MULTIPLE_CHOICE"
            else:
                questions = []
                section_type = "UNKNOWN"

            self.parsed_sections.append({
                "section_name": section_type,
                "questions": questions
            })

        return {
            "student_id": self.student_id,
            "sections": self.parsed_sections
        }

    # ==================================================
    # LOGIC 1: IDENTIFICATION & TRUE/FALSE
    # Answer BEFORE the number (handwritten or typed)
    # ==================================================
    def parse_pre_answer(self, text):
        questions = []
        # Use a regex that captures answer text preceding "1." or "1."
        # Capture lines with something like "____ 1. Question text"
        # or "True 1. Question text"
        matches = re.findall(r"([A-Za-z0-9_\-/\s]{1,80})\s+(\d+)\.", text)
        for raw_ans, num in matches:
            clean_ans = raw_ans.replace('_', '').strip()
            if not clean_ans:
                continue
            if any(tok in clean_ans.upper() for tok in ["TEST", "SECTION"]):
                continue
            try:
                questions.append({"number": int(num), "answer": clean_ans})
            except ValueError:
                continue

        # Also try another approach: lines that start with underscores then a number later
        # (helps for OCR mis-ordering)
        # Sort
        questions.sort(key=lambda x: x['number'])
        return questions

    # ==================================================
    # LOGIC 2: ENUMERATION
    # ==================================================
    def parse_enumeration(self, text):
        questions = []
        # Split by top-level numbers (1. 2. 3.)
        # We'll robustly allow "1." or "1 )" etc.
        chunks = re.split(r"(?:^|\n)\s*(\d+)\s*[.\)]\s*", text)
        # chunks => [pre, num1, text1, num2, text2, ...]
        for i in range(1, len(chunks), 2):
            qnum = chunks[i]
            block = chunks[i+1]
            try:
                qn = int(qnum)
            except:
                continue
            # find subitems: a. b. c. or a) b) c)
            sub_matches = re.findall(r"([a-zA-Z])\s*[.\)]\s*([^\n]+)", block)
            answers = []
            if sub_matches:
                answers = [m[1].strip() for m in sub_matches]
            else:
                # fallback: lines after first line are answers
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if len(lines) > 1:
                    answers = lines[1:]
            questions.append({"number": qn, "answers": answers})
        return questions

    # ==================================================
    # LOGIC 3: MULTIPLE CHOICE — Template-based OMR
    # ==================================================
    def parse_omr_with_template(self):
        results = []
        # Iterate pages in the template mapping
        for page_idx, qmap in self.MULTIPLE_CHOICE_TEMPLATE.items():
            if page_idx >= len(self.pages_images):
                continue
            img = self.pages_images[page_idx]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # For debug overlay
            overlay = img.copy()
            debug_draws = []

            for qnum, choices in qmap.items():
                # For each choice, sample mean inside circle
                filled_choices = []
                for choice_letter, (cx, cy, radius) in choices.items():
                    # ensure coordinates are integers and within bounds
                    h, w = gray.shape[:2]
                    cx = int(round(cx)); cy = int(round(cy)); radius = int(round(radius))
                    if cx < 0 or cy < 0 or cx >= w or cy >= h:
                        continue

                    mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), radius, 255, -1)
                    mean_val = cv2.mean(gray, mask=mask)[0]  # 0-255, lower => darker
                    # heuristic: lower than threshold implies filled (handwritten shading)
                    # You may want to increase threshold if handwriting is pale
                    THRESH = 170
                    is_filled = mean_val < THRESH
                    if is_filled:
                        filled_choices.append((choice_letter, mean_val))
                    # debug draw
                    color = (0, 255, 0) if is_filled else (0, 0, 255)
                    cv2.circle(overlay, (cx, cy), radius, color, 2)
                    debug_draws.append(((cx, cy), is_filled, mean_val))

                # If multiple choices flagged, choose darkest (lowest mean)
                if filled_choices:
                    filled_choices.sort(key=lambda x: x[1])  # by mean_val ascending
                    chosen = filled_choices[0][0]
                else:
                    chosen = None

                results.append({
                    "number": qnum,
                    "answer": chosen if chosen is not None else "NO_FILL_DETECTED"
                })

            # Save debug image if requested
            if self.debug and DEBUG_SAVE_IMAGES:
                try:
                    out_path = f"/tmp/omr_debug_page_{page_idx}.png"
                    # Put mean_val text
                    pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil)
                    font = ImageFont.load_default()
                    y_text = 10
                    for (cxcy, is_filled, mean_val) in debug_draws[:40]:
                        (cx, cy) = cxcy
                        draw.text((cx + 6, cy - 6), f"{int(mean_val)}", font=font)
                    pil.save(out_path)
                    print(f"[DEBUG] saved OMR overlay to {out_path}")
                except Exception as e:
                    print(f"[DEBUG] failed saving overlay: {e}")

        # If no results found from template, fallback to a simple Hough approach (optional)
        if len(results) == 0:
            # fallback: return empty list so caller may fallback to text parsing
            return []
        # sort by question number
        results.sort(key=lambda x: x["number"])
        return results


# -----------------------
# Django view
# -----------------------
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_exam_sheet(request):
    if "file" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=400)

    try:
        uploaded_file = request.FILES["file"]
        file_bytes = uploaded_file.read()
        parser = ExamParser(file_bytes, debug=True)  # debug=True saves overlays
        # You can update the template coordinates before calling extract_data:
        # parser.MULTIPLE_CHOICE_TEMPLATE[0][1]["A"] = (x, y, r)
        result_data = parser.extract_data()
        return Response(result_data, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
