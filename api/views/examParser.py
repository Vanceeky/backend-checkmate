import io
import re
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from google.cloud import vision
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

# --- CONFIGURATION ---
# Initialize Google Vision Client
# Make sure GOOGLE_APPLICATION_CREDENTIALS env var is set
client = vision.ImageAnnotatorClient()

class ExamParser:
    def __init__(self, file_bytes):
        self.file_bytes = file_bytes
        self.full_text = ""
        self.pages_images = [] # Stores CV2 images for OMR
        self.student_id = ""
        self.parsed_sections = []

    def _get_vision_text(self, pil_image):
        """Sends image to Google Vision API to get text."""
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()
        
        image = vision.Image(content=content)
        # DOCUMENT_TEXT_DETECTION is best for exams
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text

    def _pil_to_cv2(self, pil_image):
        """Helper to convert PDF2Image format to OpenCV format."""
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    def extract_data(self):
        # 1. Convert PDF to Images
        pil_images = convert_from_bytes(self.file_bytes)
        
        # 2. Process every page
        for i, img in enumerate(pil_images):
            # Store image for OMR later
            self.pages_images.append(self._pil_to_cv2(img))
            
            # Extract Text (Handwriting + Print)
            try:
                text = self._get_vision_text(img)
                self.full_text += f"\n{text}\n"
            except Exception as e:
                print(f"Error reading page {i}: {e}")

        # 3. Extract Student ID (Global Search)
        # Looks for "Student ID: 123-456" or "Student ID 123-456"
        sid_match = re.search(r"Student\s*ID[:\s]*([A-Za-z0-9\-\s]+)", self.full_text, re.IGNORECASE)
        if sid_match:
            self.student_id = sid_match.group(1).strip()

        # 4. Split Text into Logic Blocks
        # Regex looks for "Test X: Title"
        # We capture the Title and the Body until the next "Test X"
        pattern = r"(Test\s*\d+[:.]?\s*)(.*?)(?=\nTest\s*\d+|$)"
        blocks = re.findall(pattern, self.full_text, re.DOTALL | re.IGNORECASE)

        for header_prefix, block_content in blocks:
            # Determine Section Type from the Header (e.g. "Identification")
            # header_prefix might be "Test 1: IDENTIFICATION"
            # block_content is the questions
            
            # We look at the first few lines of the content or the header to guess type
            full_header = (header_prefix + block_content[:50]).upper()
            
            section_type = "UNKNOWN"
            if "IDENTIFICATION" in full_header:
                section_type = "IDENTIFICATION"
                questions = self.parse_pre_answer(block_content)
            elif "TRUE" in full_header and "FALSE" in full_header:
                section_type = "TRUE_OR_FALSE"
                questions = self.parse_pre_answer(block_content)
            elif "ENUMERATION" in full_header:
                section_type = "ENUMERATION"
                questions = self.parse_enumeration(block_content)
            elif "MULTIPLE CHOICE" in full_header:
                section_type = "MULTIPLE_CHOICE"
                # For Multiple Choice, we can try Text Parsing OR Image Parsing
                # If you want OpenCV Bubbles:
                questions = self.parse_omr_bubbles(self.pages_images)
            else:
                questions = []

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
    # Requirement: Answer is WRITTEN BEFORE the number
    # Pattern: "True 1. The CPU is..."
    # ==================================================
    def parse_pre_answer(self, text):
        questions = []
        # Regex Explanation:
        # ^ or \n  : Start of a line
        # (.+?)    : Capture the ANSWER (lazy match)
        # \s+      : Space
        # (\d+)\.  : Capture the NUMBER followed by dot
        # .* : The rest of the question text (ignored for now)
        matches = re.findall(r"(?:^|\n)(.+?)\s+\b(\d+)\.", text)
        
        for raw_ans, num in matches:
            # Clean the answer (remove underscores lines like _____)
            clean_ans = raw_ans.replace('_', '').strip()
            # If the OCR picked up "Test 1" as an answer, ignore it
            if "Test" in clean_ans or "Section" in clean_ans:
                continue
                
            questions.append({
                "number": int(num),
                "answer": clean_ans
            })
        
        # Sort by number to be safe
        questions.sort(key=lambda x: x['number'])
        return questions

    # ==================================================
    # LOGIC 2: ENUMERATION
    # Requirement: 1. Question -> a. Ans, b. Ans
    # ==================================================
    def parse_enumeration(self, text):
        questions = []
        
        # Split text into chunks starting with a number "1."
        # This regex looks for a digit+dot at the start of a line
        chunks = re.split(r"(?:^|\n)(\d+)\.", text)
        
        # re.split returns [pre_text, num1, text1, num2, text2...]
        # We iterate in steps of 2
        for i in range(1, len(chunks), 2):
            q_num = int(chunks[i])
            q_text_block = chunks[i+1]
            
            # Find sub-items (a. b. c.) inside this block
            # Looking for single letter + dot
            sub_matches = re.findall(r"\b([a-zA-Z])\.\s*(.+)", q_text_block)
            
            answers = []
            if sub_matches:
                for letter, val in sub_matches:
                    answers.append(val.strip())
            else:
                # If regex fails, assume every non-empty line is an answer
                lines = [line.strip() for line in q_text_block.split('\n') if line.strip()]
                # Heuristic: Remove the question text itself if it's the first line
                if len(lines) > 0:
                    answers = lines[1:] # Skipping first line assuming it's the question
            
            questions.append({
                "number": q_num,
                "answers": answers
            })
            
        return questions

    # ==================================================
    # LOGIC 3: MULTIPLE CHOICE (OMR / BUBBLES)
    # Requirement: Detect filled circles using OpenCV
    # ==================================================
    def parse_omr_bubbles(self, cv2_images):
        detected_answers = []
        
        # Flatten all pages to find bubbles (simplest approach for non-fixed layout)
        # Ideally, you'd limit this to the page where "Test 4" was detected.
        
        global_bubble_list = []

        for img in cv2_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # HoughCircles to find bubbles
            # TUNING REQUIRED: minRadius and maxRadius depend on scanning DPI
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                       param1=50, param2=30, minRadius=10, maxRadius=25)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Check which are filled
                for (x, y, r) in circles:
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.circle(mask, (x, y), r, 255, -1)
                    mean_val = cv2.mean(gray, mask=mask)[0]
                    
                    # Darker = Filled. Threshold < 150 usually works for black ink
                    if mean_val < 180: 
                        # This is a filled bubble!
                        global_bubble_list.append((x, y))

        # LOGIC MAPPING:
        # This part is tricky without fixed coordinates. 
        # For this code, I will return the COUNT of filled bubbles found 
        # or a placeholder, because mapping X/Y to Question 1/A requires 
        # knowing exactly where Q1 is printed on the page.
        
        # For CheckMate prototype, we will return a special flag so you can 
        # see it worked, but mapping requires a template grid.
        
        # Mocking the response based on found bubbles
        # In a real app, you sort these (x,y) by Y-coordinate to get Q1, Q2...
        if len(global_bubble_list) > 0:
             # Sort by Y (top to bottom)
             global_bubble_list.sort(key=lambda k: k[1]) 
             
             for i, bubble in enumerate(global_bubble_list):
                 detected_answers.append({
                     "number": i + 1,
                     "answer": "BUBBLE_FILLED", 
                     "coordinates": bubble
                 })
        else:
            # Fallback: If no bubbles found, maybe they wrote the letter?
            # We would fallback to text parsing here if needed.
            pass

        return detected_answers

# --- DJANGO VIEW ---

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_exam_sheet(request):
    if "file" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=400)

    try:
        uploaded_file = request.FILES["file"]
        file_bytes = uploaded_file.read()
        
        # Initialize the Parser Engine
        parser = ExamParser(file_bytes)
        
        # Run extraction
        result_data = parser.extract_data()
        
        return Response(result_data, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)