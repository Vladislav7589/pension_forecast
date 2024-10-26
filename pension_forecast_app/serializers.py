from rest_framework import serializers


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
