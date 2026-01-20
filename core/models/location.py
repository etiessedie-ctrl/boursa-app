from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class UserLocation(models.Model):
    """
    Model to store user locations for real-time cartography
    """
    device_id = models.CharField(max_length=100, unique=True, help_text="Unique device identifier")
    latitude = models.FloatField(help_text="Latitude coordinate")
    longitude = models.FloatField(help_text="Longitude coordinate")
    accuracy = models.FloatField(null=True, blank=True, help_text="Location accuracy in meters")
    timestamp = models.DateTimeField(default=timezone.now, help_text="When this location was recorded")
    is_active = models.BooleanField(default=True, help_text="Whether this user is currently active")
    last_seen = models.DateTimeField(default=timezone.now, help_text="Last time this user was seen")

    class Meta:
        ordering = ['-last_seen']
        indexes = [
            models.Index(fields=['device_id']),
            models.Index(fields=['last_seen']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return f"Location for {self.device_id} at ({self.latitude}, {self.longitude})"

    def update_last_seen(self):
        """Update the last seen timestamp"""
        self.last_seen = timezone.now()
        self.save(update_fields=['last_seen'])

    @classmethod
    def get_active_locations(cls, max_age_minutes=10):
        """
        Get locations that have been seen within the last max_age_minutes
        """
        from django.utils import timezone
        from datetime import timedelta

        cutoff_time = timezone.now() - timedelta(minutes=max_age_minutes)
        return cls.objects.filter(
            is_active=True,
            last_seen__gte=cutoff_time
        ).order_by('-last_seen')