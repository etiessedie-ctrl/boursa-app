from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class TestHistory(models.Model):
    """Model to store test execution history"""

    TEST_TYPES = [
        ('wilcoxon', 'Test de Wilcoxon'),
        ('mannwhitney', 'Test de Mann-Whitney'),
        ('kruskal', 'Test de Kruskal-Wallis'),
        ('spearman', 'Corrélation de Spearman'),
        ('friedman', 'Test de Friedman'),
        ('kolmogorov_smirnov', 'Test de Kolmogorov-Smirnov'),
        ('shapiro_wilk', 'Test de Shapiro-Wilk'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    test_type = models.CharField(max_length=50, choices=TEST_TYPES)
    test_name = models.CharField(max_length=100)
    selected_columns = models.JSONField()  # List of column names
    filename = models.CharField(max_length=255, null=True, blank=True)

    # Test results
    statistic = models.FloatField(null=True, blank=True)
    p_value = models.FloatField(null=True, blank=True)
    result_data = models.JSONField()  # Complete result dictionary

    # Additional fields from existing table
    interpretation = models.TextField(null=True, blank=True)
    histogram = models.CharField(max_length=100, null=True, blank=True)
    qqplot = models.CharField(max_length=100, null=True, blank=True)
    timeseries = models.CharField(max_length=100, null=True, blank=True)
    data_file_id = models.BigIntegerField(null=True, blank=True)

    # Metadata
    timestamp = models.DateTimeField(default=timezone.now)
    execution_time = models.FloatField(null=True, blank=True)  # in seconds
    alpha = models.FloatField(default=0.05)
    sample_size = models.IntegerField(null=True, blank=True)

    # Significance
    is_significant = models.BooleanField(default=False)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Historique de test"
        verbose_name_plural = "Historiques de tests"

    def __str__(self):
        return f"{self.test_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    def save(self, *args, **kwargs):
        # Determine if result is significant
        if self.p_value is not None:
            self.is_significant = self.p_value < self.alpha
        super().save(*args, **kwargs)

    @property
    def significance_status(self):
        """Return significance status as string"""
        if self.p_value is None:
            return "N/A"
        return "Significatif" if self.is_significant else "Non significatif"

    @property
    def test_category(self):
        """Return test category for styling"""
        categories = {
            'kolmogorov_smirnov': 'normalite',
            'shapiro_wilk': 'normalite',
            'wilcoxon': 'non-param',
            'mannwhitney': 'non-param',
            'kruskal': 'non-param',
            'friedman': 'non-param',
            'spearman': 'correlation',
        }
        return categories.get(self.test_type, 'other')