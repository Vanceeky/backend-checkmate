from rest_framework import serializers
from api.models import ExamSubmission, AnswerKeySection

class ExamSubmissionSerializer(serializers.ModelSerializer):
    examTitle = serializers.CharField(source='exam.id', read_only=True)
    examType = serializers.CharField(source='exam.exam_type', read_only=True)
    subject = serializers.CharField(source='exam.subject', read_only=True)
    semester = serializers.CharField(source='exam.semester', read_only=True)
    schoolYear = serializers.CharField(source='exam.school_year', read_only=True)
    pdfUrl = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    submitted_at = serializers.DateTimeField()
    breakdown = serializers.JSONField(source='section_scores', read_only=True)

    class Meta:
        model = ExamSubmission
        fields = [
            'examTitle',
            'examType',
            'subject',
            'semester',
            'schoolYear',
            'pdfUrl',
            'status',
            'score',
            'submitted_at',
            'breakdown',
        ]

    def get_status(self, obj):
        total_points = obj.exam.total_exam_points
        passing_score = total_points * 0.75

        if obj.score is None:
            return "Pending"

        return "Passed" if obj.score >= passing_score else "Failed"
    def get_pdfUrl(self, obj):
        request = self.context.get('request')
        if obj.scanned_sheet:
            return request.build_absolute_uri(obj.scanned_sheet.url)
        return None



class SectionScoreSerializer(serializers.ModelSerializer):
    sectionTitle = serializers.CharField(source="section_type")
    total = serializers.IntegerField(source="total_points")
    
    class Meta:
        model = AnswerKeySection
        fields = ["sectionTitle", "score", "total"]

class ExamSubmissionDetailSerializer(serializers.ModelSerializer):
    examId = serializers.CharField(source="exam.id")
    title = serializers.CharField(source="exam.exam_type")
    subject = serializers.CharField(source="exam.subject")
    semester = serializers.CharField(source="exam.semester")
    schoolYear = serializers.CharField(source="exam.school_year")
    totalPoints = serializers.IntegerField(source="exam.total_exam_points")
    percentage = serializers.FloatField(source="exam.passing_percentage")

    status = serializers.SerializerMethodField()
    submitted_at = serializers.DateTimeField()
    pdfUrl = serializers.SerializerMethodField()
    breakdown = serializers.SerializerMethodField()

    class Meta:
        model = ExamSubmission
        fields = [
            "examId",
            "title",
            "subject",
            "semester",
            "schoolYear",
            "totalPoints",
            "percentage",
            "pdfUrl",
            "score",
            "status",
            "submitted_at",
            "breakdown",
        ]

    def get_status(self, obj):
        total_points = obj.exam.total_exam_points
        passing_score = total_points * 0.75

        if obj.score is None:
            return "Pending"

        return "Passed" if obj.score >= passing_score else "Failed"

    def get_pdfUrl(self, obj):
        if obj.scanned_sheet:
            request = self.context.get("request")
            return request.build_absolute_uri(obj.scanned_sheet.url)
        return None

    def get_breakdown(self, obj):
        # Get all sections for this exam
        sections = obj.exam.answer_key_sections.all()
        result = []
        for section in sections:
            # Example: compute proportional score
            # Here you might want to store actual per-section scores in another model
            result.append({
                "sectionTitle": section.section_type,
                "score": section.total_points,  # Replace if you store actual score
                "total": section.total_points,
            })
        return result
