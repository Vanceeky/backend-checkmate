from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from api.models import ExamSubmission
from api.serializers.student import ExamSubmissionSerializer, ExamSubmissionDetailSerializer


class StudentExamDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, exam_id):
        student = request.user.student  # get the logged-in student
        submission = get_object_or_404(ExamSubmission, exam__id=exam_id, student=student)
        serializer = ExamSubmissionSerializer(submission, context={'request': request})
        return Response(serializer.data, status=200)


class StudentExamSubmissionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Assuming request.user is linked to Student
        student = getattr(request.user, "student", None)
        if not student:
            return Response({"detail": "Student not found."}, status=404)

        submissions = ExamSubmission.objects.filter(student=student).order_by("-submitted_at")
        serializer = ExamSubmissionDetailSerializer(submissions, many=True, context={"request": request})
        return Response(serializer.data, status=200)