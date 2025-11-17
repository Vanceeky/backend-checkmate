from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models import JSONField

class CustomUser(AbstractUser):
    
    class Role(models.TextChoices):
        ADMIN = 'admin', 'ADMIN'
        STUDENT = 'student', 'STUDENT'
        INSTRUCTOR = 'instructor', 'INSTRUCTOR'

    role = models.CharField(max_length=20, choices=Role.choices)

    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)

    def __str__(self):
        return f"{self.get_full_name().upper()} - {self.role.upper()}"
    

class Instructor(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='instructor')
    institution = models.CharField(max_length=100, null=False, blank=False)
    instructor_id = models.ImageField(upload_to='instructors/ids/', null=False, blank=False)

    def __str__(self):
        return f"{self.institution.upper()} - {self.user.get_full_name().upper()}"
    

class Student(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='student')
    institution = models.CharField(max_length=100, null=False, blank=False)
    student_id = models.ImageField(upload_to='students/ids/', null=False, blank=False)

    def __str__(self):
        return f"{self.institution.upper()} -{self.user.get_full_name().upper()}"
    


class Exam(models.Model):
    id = models.CharField(primary_key=True, max_length=255, editable=False)  

    exam_type = models.CharField(max_length=100, null=False, blank=False)
    semester = models.CharField(max_length=100, null=False, blank=False)
    subject = models.CharField(max_length=100, null=False, blank=False)
    school_year = models.CharField(max_length=100, null=False, blank=False)

    total_exam_points = models.IntegerField(default=0, null=False, blank=False)
    passing_percentage = models.FloatField(default=75.0, null=False, blank=False)


    created_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, related_name='created_exams')
    created_at = models.DateTimeField(auto_now_add=True)

    def passing_score(self):
        return round(self.total_exam_points * self.passing_percentage / 100)
    
    def save(self, *args, **kwargs):
        if not self.id:
            # Create descriptive ID (PRIMARY KEY)
            self.id = (
                f"{self.exam_type.upper()}-"
                f"{self.subject.upper()}-"
                f"{self.school_year.replace(' ', '')}"
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return self.id
    

class AnswerKeySection(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='answer_key_sections')
    section_type = models.CharField(max_length=100, null=False, blank=False)
    total_points = models.IntegerField(default=0, null=False, blank=False)

    items = JSONField(default = list,null=False, blank=False)

    def __str__(self):
        return f"{self.exam} - {self.section_type.upper()} - {self.total_points}"
    




class ExamSubmission(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="submissions")
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='submissions')
    score = models.IntegerField(null=False, blank=False)

    status = models.BooleanField(default=False)
    scanned_sheet = models.FileField(upload_to="exam_submissions/", null=True, blank=True)
    
    # NEW FIELD
    section_scores = models.JSONField(
        default=list, 
        blank=True,
        help_text="List of per-section scores: [{'sectionTitle': 'MCQ', 'score': 10, 'total': 15}]"
    )


    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("exam", "student")  # Prevent duplicate submissions

    def save(self, *args, **kwargs):
        # Auto calculate pass/fail if not manually set
        if not self.status:
            self.status = self.score >= self.exam.passing_score()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.student.user.get_full_name()} - {self.exam.id}"
