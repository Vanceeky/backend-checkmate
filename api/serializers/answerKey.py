from rest_framework import serializers

class AnswerKeySectionSerializer(serializers.Serializer):
    sectionType = serializers.CharField()
    totalPoints = serializers.IntegerField()
    items = serializers.JSONField()