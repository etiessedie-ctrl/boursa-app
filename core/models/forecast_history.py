from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class ForecastHistory(models.Model):
    """Model to store forecast execution history"""

    MODEL_TYPES = [
        ('arima', 'ARIMA'),
        ('sarima', 'SARIMA'),
        ('prophet', 'Prophet'),
        ('lstm', 'LSTM'),
        ('xgboost', 'XGBoost'),
        ('linear', 'Régression Linéaire'),
        ('random_forest', 'Random Forest'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    model_name = models.CharField(max_length=50, choices=MODEL_TYPES)
    target_column = models.CharField(max_length=100)
    filename = models.CharField(max_length=255, null=True, blank=True)

    # Forecast parameters
    forecast_steps = models.IntegerField()
    forecast_type = models.CharField(max_length=50, null=True, blank=True)
    forecast_interval = models.CharField(max_length=50, null=True, blank=True)

    # Results - matching existing table
    forecast_values = models.TextField(null=True, blank=True)  # JSON string of predicted values
    confidence_intervals = models.TextField(null=True, blank=True)  # JSON string
    metrics = models.TextField(null=True, blank=True)  # JSON string

    # Additional fields from existing table
    model = models.CharField(max_length=100, null=True, blank=True)  # For compatibility
    confidence_level = models.FloatField(null=True, blank=True)
    forecast_plot = models.CharField(max_length=100, null=True, blank=True)
    data_file_id = models.BigIntegerField(null=True, blank=True)

    # Metadata
    timestamp = models.DateTimeField(default=timezone.now)
    execution_time = models.FloatField(null=True, blank=True)  # in seconds

    # Status
    success = models.BooleanField(default=True)
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Historique de prévision"
        verbose_name_plural = "Historiques de prévisions"

    def __str__(self):
        return f"{self.model_name} - {self.target_column} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    @property
    def forecast_range(self):
        """Return forecast date range"""
        if self.forecast_values:
            try:
                values = json.loads(self.forecast_values)
                return f"1-{len(values)} périodes"
            except:
                return "N/A"
        return "N/A"

    @property
    def predictions(self):
        """Return predictions as list for template compatibility"""
        if self.forecast_values:
            try:
                return json.loads(self.forecast_values)
            except:
                return []
        return []