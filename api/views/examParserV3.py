# upgraded_exam_parser.py
import io
import os
import re
import json
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# PDF rendering: try fitz (PyMuPDF) then pdf2image as fallback
def render_pdf_first_page_bytes(pdf_bytes, dpi=200):
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)  # increase resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception:
        from pdf2image import convert_from_bytes
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        return imgs[0]

# OMR auto-detection utilities (best-effort)
def detect_bubbles_auto(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 10000:
            continue
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        radius = float(radius)
        if radius < 6 or radius > 90:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue
        candidates.append((int(x), int(y), int(radius), float(circularity), int(area)))
    # sort and dedupe close centers
    candidates_sorted = sorted(candidates, key=lambda c: (c[1], c[0]))
    dedup = []
    for c in candidates_sorted:
        if not dedup:
            dedup.append(c)
        else:
            prev = dedup[-1]
            if abs(prev[0]-c[0]) < 6 and abs(prev[1]-c[1]) < 6:
                if c[2] > prev[2]:
                    dedup[-1] = c
            else:
                dedup.append(c)
    return dedup, img, gray, th

def cluster_rows(circles, page_height, y_tol=None):
    if not circles:
        return []
    if y_tol is None:
        y_tol = max(8, int(0.02 * page_height))
    sorted_c = sorted(circles, key=lambda c: (c[1], c[0]))
    rows = []
    current_row = [sorted_c[0]]
    for c in sorted_c[1:]:
        if abs(c[1] - current_row[-1][1]) <= y_tol:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    if current_row:
        rows.append(current_row)
    rows_sorted = [sorted(r, key=lambda e: e[0]) for r in rows]
    return rows_sorted

def build_template_from_rows(rows, start_q=1):
    template_page = {}
    qnum = start_q
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for row in rows:
        choice_map = {}
        for i, (x,y,r,circ,area) in enumerate(row):
            letter = letters[i] if i < len(letters) else f"opt{i}"
            choice_map[letter] = (int(x), int(y), int(r))
        template_page[qnum] = choice_map
        qnum += 1
    return template_page

# ---------- OCR engine handling ----------
def create_vision_client_if_possible():
    # if GOOGLE_APPLICATION_CREDENTIALS env var exists and file present, try to create client
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.exists(env_path):
        try:
            from google.oauth2 import service_account
            from google.cloud import vision
            creds = service_account.Credentials.from_service_account_file(env_path)
            return vision.ImageAnnotatorClient(credentials=creds), "google_vision"
        except Exception as e:
            print("[OCR] failed to create google vision client:", e)
            return None, None
    # try application default (may still work in GCP environments)
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        return client, "google_vision"
    except Exception:
        return None, None

def create_tesseract_if_possible():
    try:
        import pytesseract
        return pytesseract, "tesseract"
    except Exception:
        return None, None

# ---------- The upgraded ExamParser ----------
class ExamParser:
    def __init__(self, file_bytes, debug=False, template=None):
        """
        file_bytes: raw PDF bytes
        debug: if True, will save debug overlay to /mnt/data/omr_debug_page_0.png and print info
        template: optional dict { page_index: { question_number: { "A": (x,y,r), ... } } }
                  If None, parser will attempt auto-calibration from the first page of the PDF.
        """
        self.file_bytes = file_bytes
        self.full_text = ""
        self.pages_images = []
        self.student_id = ""
        self.parsed_sections = []
        self.debug = debug
        self.debug_path = "/mnt/data/omr_debug_page_0.png"
        self.ocr_engine_used = "none"

        # user-provided or auto-generated template
        self.MULTIPLE_CHOICE_TEMPLATE = template if template else {}

        # create OCR clients (detect what's available)
        self.vision_client, vision_label = create_vision_client_if_possible()
        self.pytesseract, tess_label = create_tesseract_if_possible()
        # choose preferred engine: google -> tesseract -> none
        if self.vision_client:
            self.ocr_engine_used = vision_label
        elif self.pytesseract:
            self.ocr_engine_used = tess_label
        else:
            self.ocr_engine_used = "none"

    # OCR wrapper (uses chosen engine)
    def _get_vision_text(self, pil_image):
        # prefer Google Vision if available (better handwriting support)
        if self.vision_client:
            try:
                from google.cloud import vision
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                content = img_byte_arr.getvalue()
                image = vision.Image(content=content)
                response = self.vision_client.document_text_detection(image=image)
                if response.error.message:
                    # raise or fallback
                    raise Exception(response.error.message)
                return response.full_text_annotation.text
            except Exception as e:
                print("[OCR] Google Vision failed:", e)
                # fallback to tesseract if available
        if self.pytesseract:
            try:
                # grayscale -> tesseract
                img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
                text = self.pytesseract.image_to_string(img, config="--psm 6")
                return text
            except Exception as e:
                print("[OCR] Tesseract failed:", e)
                return ""
        # no OCR available
        return ""

    def _pil_to_cv2(self, pil_image):
        open_cv_image = np.array(pil_image)
        return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    def auto_calibrate_template_from_pdf(self):
        """
        If MULTIPLE_CHOICE_TEMPLATE is empty, attempt automatic detection of bubble locations
        from the first page of the PDF. This is best-effort and must be visually verified.
        """
        if self.MULTIPLE_CHOICE_TEMPLATE:
            return  # already provided

        try:
            pil = render_pdf_first_page_bytes(self.file_bytes)
            circles, img, gray, th = detect_bubbles_auto(pil)
            rows = cluster_rows(circles, pil.size[1], y_tol=max(8, int(0.02 * pil.size[1])))
            page_map = build_template_from_rows(rows, start_q=1)
            if page_map:
                self.MULTIPLE_CHOICE_TEMPLATE[0] = page_map
                # save debug overlay
                overlay = img.copy()
                for row_idx, row in enumerate(rows):
                    for col_idx, (x,y,r,circ,area) in enumerate(row):
                        cv2.circle(overlay, (x,y), r, (0,255,0), 2)
                        cv2.putText(overlay, f"Q{row_idx+1}-{col_idx+1}", (x+6, y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                # write file
                cv2.imwrite(self.debug_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if self.debug:
                    print(f"[OMR] Auto-calibrated template saved. Debug overlay: {self.debug_path}")
            else:
                if self.debug:
                    print("[OMR] No template detected during auto-calibration.")
        except Exception as e:
            if self.debug:
                print("[OMR] auto_calibrate_template_from_pdf failed:", e)

    # core extraction
    def extract_data(self):
        # convert PDF to images
        try:
            # use our render helper (returns PIL)
            pil_imgs = [render_pdf_first_page_bytes(self.file_bytes)]  # we only need first page for now
        except Exception as e:
            raise Exception(f"Failed to render PDF: {e}")

        for i, pil_img in enumerate(pil_imgs):
            self.pages_images.append(self._pil_to_cv2(pil_img))
            try:
                text = self._get_vision_text(pil_img)
                self.full_text += f"\n{text}\n"
            except Exception as e:
                print(f"[extract_data] page {i} OCR error: {e}")

        # student id detection
        sid_match = re.search(r"Student\s*ID\s*[:\-]?\s*([A-Za-z0-9\-\s]+)", self.full_text, re.IGNORECASE)
        if sid_match:
            self.student_id = sid_match.group(1).strip()

        # attempt auto-calibration if no template provided
        if not self.MULTIPLE_CHOICE_TEMPLATE:
            self.auto_calibrate_template_from_pdf()

        # split into blocks (robust)
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
            "sections": self.parsed_sections,
            "ocr_engine_used": self.ocr_engine_used,
            "omr_debug_image": self.debug_path if (self.debug and os.path.exists(self.debug_path)) else None,
            "template_used": self.MULTIPLE_CHOICE_TEMPLATE
        }

    # Identification / True-False (answer before number)
    def parse_pre_answer(self, text):
        questions = []
        matches = re.findall(r"([A-Za-z0-9_\-/\s]{1,120})\s+(\d+)\.", text)
        for raw_ans, num in matches:
            clean = raw_ans.replace('_','').strip()
            if not clean:
                continue
            if any(tok in clean.upper() for tok in ["TEST","SECTION"]):
                continue
            try:
                questions.append({"number": int(num), "answer": clean})
            except:
                continue
        questions.sort(key=lambda x: x['number'])
        return questions

    # Enumeration
    def parse_enumeration(self, text):
        questions = []
        chunks = re.split(r"(?:^|\n)\s*(\d+)\s*[.\)]\s*", text)
        for i in range(1, len(chunks), 2):
            qnum = chunks[i]
            block = chunks[i+1]
            try:
                qn = int(qnum)
            except:
                continue
            sub_matches = re.findall(r"([a-zA-Z])\s*[.\)]\s*([^\n]+)", block)
            answers = []
            if sub_matches:
                answers = [m[1].strip() for m in sub_matches]
            else:
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if len(lines) > 1:
                    answers = lines[1:]
            questions.append({"number": qn, "answers": answers})
        return questions

    # OMR using template (page-based)
    def parse_omr_with_template(self):
        detected = []
        for page_idx, mapping in self.MULTIPLE_CHOICE_TEMPLATE.items():
            if page_idx >= len(self.pages_images):
                continue
            img = self.pages_images[page_idx]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for qnum, opts in mapping.items():
                chosen = None
                chosen_mean = None
                for letter, (cx, cy, r) in opts.items():
                    h, w = gray.shape[:2]
                    cx = int(round(cx)); cy = int(round(cy)); r = int(round(r))
                    if cx < 0 or cy < 0 or cx >= w or cy >= h:
                        continue
                    mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), r, 255, -1)
                    mean_val = cv2.mean(gray, mask=mask)[0]
                    # threshold heuristic (adjust if required)
                    THRESH = 170
                    is_filled = mean_val < THRESH
                    if is_filled:
                        if chosen is None or mean_val < chosen_mean:
                            chosen = letter
                            chosen_mean = mean_val
                detected.append({"number": qnum, "answer": chosen if chosen else "NO_FILL_DETECTED"})
        detected.sort(key=lambda x: x["number"])
        return detected

# --------- Example Django view (drop-in) ----------
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_exam_sheet(request):
    if "file" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=400)
    try:
        uploaded_file = request.FILES["file"]
        file_bytes = uploaded_file.read()

        # Optional: If you already want to supply the exact template (after calibration),
        # build the dict and pass it into ExamParser(..., template=YOUR_DICT)
        parser = ExamParser(file_bytes, debug=True, template=None)
        result = parser.extract_data()
        return Response(result, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
