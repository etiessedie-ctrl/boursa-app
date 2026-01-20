from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class DataFile(models.Model):
    """Model to store uploaded data files"""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()

    # File metadata
    columns = models.JSONField()  # List of column names
    row_count = models.IntegerField()
    file_type = models.CharField(max_length=10)  # csv, xlsx, etc.

    # Upload info
    uploaded_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Fichier de données"
        verbose_name_plural = "Fichiers de données"

    def __str__(self):
        return f"{self.original_filename} ({self.user.username})"


class Dataset(models.Model):
    """Model to store processed datasets"""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Data
    data = models.JSONField()  # Processed data as JSON
    columns = models.JSONField()  # Column information
    row_count = models.IntegerField()

    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    # Source
    source_file = models.ForeignKey(DataFile, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Jeu de données"
        verbose_name_plural = "Jeux de données"

    def __str__(self):
        return f"{self.name} ({self.user.username})"