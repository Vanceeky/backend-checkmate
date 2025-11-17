from rest_framework import serializers
from api.models import Exam, AnswerKeySection

from api.serializers.answerKey import AnswerKeySectionSerializer


class BasicInfoSerializer(serializers.Serializer):
    examType = serializers.CharField()
    semester = serializers.CharField()
    subject = serializers.CharField()
    schoolYear = serializers.CharField()


class ExamCreateSerializer(serializers.Serializer):
    basicInfo = BasicInfoSerializer()
    answerKey = AnswerKeySectionSerializer(many=True)
    totalExamPoints = serializers.IntegerField()

    def validate(self, data):
        basic = data['basicInfo']
        required = ['examType', 'subject', 'schoolYear', 'semester']

        for field in required:
            if field not in basic:
                raise serializers.ValidationError(
                    f"{field} is required"
                )
            
        return data
    

    def create(self, validated_data):
        request = self.context['request']
     
        
        basic = validated_data['basicInfo']

        exam_id = (
            f"{basic['examType'].upper()}-"
            f"{basic['subject'].upper()}-"
            f"{basic['schoolYear'].replace(' ', '')}"
        )

        if Exam.objects.filter(id=exam_id).exists():
            raise serializers.ValidationError(
                f"{exam_id} already exists"
            )
        
        exam = Exam.objects.create(
            id=exam_id,
            exam_type=basic['examType'],
            subject=basic['subject'],
            school_year=basic['schoolYear'],
            semester=basic['semester'],
            total_exam_points=validated_data['totalExamPoints'],
            created_by=request.user
        )

        for answer_key in validated_data['answerKey']:
            AnswerKeySection.objects.create(
                exam=exam,
                section_type=answer_key['sectionType'],
                total_points=answer_key['totalPoints'],
                items=answer_key['items']
            )

        return exam