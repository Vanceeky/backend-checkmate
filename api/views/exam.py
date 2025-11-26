from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from api.models import Exam

from api.serializers.exam import ExamCreateSerializer

import re

from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response

from rest_framework.parsers import MultiPartParser, FormParser

from django.http import JsonResponse
from api.checkmate_ocr import process_exam_pdf



class CreateExamView(APIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ExamCreateSerializer

    def post(self, request):
        serializer = ExamCreateSerializer(data = request.data, context={'request': request})

        if serializer.is_valid():
            exam = serializer.save()

            return Response({
                "message": "Exam created successfully.",
                "examId": exam.id,
                "passingScore": exam.passing_score(),
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class InstructorExamListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user  # instructor (CustomUser)
        exams = Exam.objects.filter(created_by=user).order_by('created_at')

        result = []

        for exam in exams:
            submissions = exam.submissions.select_related('student', 'student__user').all()

            student_results = []
            for sub in submissions:
                student_results.append({
                    "name": sub.student.user.get_full_name(),
                    "score": sub.score,
                    "total": exam.total_exam_points,
                    "pdfUrl": sub.scanned_sheet.url if sub.scanned_sheet else None,
                    "studentId": sub.student.id,  # or sub.student.user.id
                })

            result.append({
                "examTitle": exam.id,  # value = exam ID
                "examId": exam.id,
                "subject": exam.subject,
                "semester": exam.semester,
                "schoolYear": exam.school_year,
                "examType": exam.exam_type,
                "hasScannedSheets": any(sub.scanned_sheet for sub in submissions),
                "studentResults": student_results,
            })

        return Response(result, status=200)
    


def parse_exam_sheet(request):
    if request.method == "POST" and request.FILES.get("exam_pdf"):
        pdf_file = request.FILES["exam_pdf"]
        pdf_bytes = pdf_file.read()
        result = process_exam_pdf(pdf_bytes)
        return JsonResponse(result)
    return JsonResponse({"error": "No file uploaded"}, status=400)