from django.urls import path
from api.views.auth import CustomTokenObtainPairView

from api.views.exam import CreateExamView, InstructorExamListView
from api.views.student import StudentExamDetailView, StudentExamSubmissionsView


urlpatterns = [
    path('auth/jwt/create/', CustomTokenObtainPairView.as_view(), name='jwt-create'),
    path("exams/create/", CreateExamView.as_view(), name="create-exam"),
    path('exams/instructor/', InstructorExamListView.as_view(), name='instructor-exams'),

    path("exams/student/", StudentExamSubmissionsView.as_view(), name="student-exams"),
    path('exams/student/<str:exam_id>/', StudentExamDetailView.as_view(), name='student-exam-detail'),
]
