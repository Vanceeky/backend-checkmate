from django.contrib import admin
from .models import CustomUser, Student, Instructor, Exam, AnswerKeySection, ExamSubmission
# Register your models here.

admin.site.register(CustomUser)
admin.site.register(Student)
admin.site.register(Instructor)


class AnswerKeySectionInline(admin.TabularInline):   # <--- Tabular!!!
    model = AnswerKeySection
    extra = 1
    fields = ("section_type", "total_points", "items")
    show_change_link = True


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "exam_type",
        "subject",
        "school_year",
        "total_exam_points",
        "passing_percentage",
        "created_by",
        "created_at"
    )

    search_fields = ("id", "subject", "exam_type", "school_year")
    list_filter = ("exam_type", "semester", "school_year")

    inlines = [AnswerKeySectionInline]    # <--- Adds tabular sections inside Exam admin


""" @admin.register(AnswerKeySection)
class AnswerKeySectionAdmin(admin.ModelAdmin):
    list_display = ("section_type", "exam", "total_points")
    search_fields = ("section_type",) """


@admin.register(ExamSubmission)
class ExamSubmissionAdmin(admin.ModelAdmin):
    list_display = ('exam', 'student_name', 'score')

    def student_name(self, obj):
        return obj.student.user.get_full_name()
    student_name.short_description = 'Student Name'