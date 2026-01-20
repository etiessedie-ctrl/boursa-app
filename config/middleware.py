"""
Security middleware for production deployment.
Adds additional security headers and protections.
"""

from django.conf import settings
from django.http import HttpResponse


class SecurityHeadersMiddleware:
    """
    Middleware to add security headers to all responses.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Only add security headers in production
        if not settings.DEBUG:
            # Prevent clickjacking
            response['X-Frame-Options'] = 'DENY'

            # Prevent MIME type sniffing
            response['X-Content-Type-Options'] = 'nosniff'

            # Enable XSS filtering
            response['X-XSS-Protection'] = '1; mode=block'

            # Referrer Policy
            response['Referrer-Policy'] = 'strict-origin-when-cross-origin'

            # Permissions Policy (formerly Feature Policy)
            response['Permissions-Policy'] = (
                'geolocation=(), microphone=(), camera=(), '
                'magnetometer=(), gyroscope=(), accelerometer=(), '
                'payment=(), usb=()'
            )

            # Cross-Origin policies
            response['Cross-Origin-Embedder-Policy'] = 'require-corp'
            response['Cross-Origin-Opener-Policy'] = 'same-origin'
            response['Cross-Origin-Resource-Policy'] = 'same-origin'

            # Remove server header
            if 'Server' in response:
                del response['Server']

            # Remove X-Powered-By header if present
            if 'X-Powered-By' in response:
                del response['X-Powered-By']

        return response


class ContentSecurityPolicyMiddleware:
    """
    Middleware to add Content Security Policy headers.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Only add CSP in production
        if not settings.DEBUG:
            csp_parts = []

            # Default policy
            csp_parts.append("default-src 'self'")

            # Style sources
            csp_parts.append("style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net")

            # Script sources
            csp_parts.append("script-src 'self' 'unsafe-inline' https://code.jquery.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com")

            # Font sources
            csp_parts.append("font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net")

            # Image sources
            csp_parts.append("img-src 'self' data: https:")

            # Connect sources (for AJAX/fetch requests)
            csp_parts.append("connect-src 'self' https://api.example.com")  # Adjust for your APIs

            # Frame sources
            csp_parts.append("frame-src 'none'")

            # Object sources
            csp_parts.append("object-src 'none'")

            # Base URI
            csp_parts.append("base-uri 'self'")

            # Form actions
            csp_parts.append("form-action 'self'")

            response['Content-Security-Policy'] = '; '.join(csp_parts)

        return response