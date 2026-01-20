from .base import *
import os

# Importer les middlewares personnalisés
try:
    from config.middleware import SecurityHeadersMiddleware, ContentSecurityPolicyMiddleware
except ImportError:
    # Fallback si le middleware n'existe pas
    SecurityHeadersMiddleware = None
    ContentSecurityPolicyMiddleware = None

DEBUG = False

ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1', cast=lambda v: [s.strip() for s in str(v).split(',') if s.strip()])

# Security settings for production
SECRET_KEY = config('SECRET_KEY')  # Must be set in environment

# Add security middlewares if available
if SecurityHeadersMiddleware and ContentSecurityPolicyMiddleware:
    MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'whitenoise.middleware.WhiteNoiseMiddleware',  # Pour servir les fichiers statiques sur cloud
        'config.middleware.SecurityHeadersMiddleware',
        'config.middleware.ContentSecurityPolicyMiddleware',
    ] + MIDDLEWARE
else:
    MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'whitenoise.middleware.WhiteNoiseMiddleware',  # Pour servir les fichiers statiques sur cloud
    ] + MIDDLEWARE

# Database configuration - Supporte MySQL (local) et PostgreSQL (cloud)
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Configuration pour hébergeurs cloud (PostgreSQL)
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.config(
            default=DATABASE_URL,
            conn_max_age=600
        )
    }
else:
    # Configuration MySQL pour déploiement local
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': config('DB_NAME', default='boursa_db'),
            'USER': config('DB_USER', default='boursa_user'),
            'PASSWORD': config('DB_PASSWORD', default=''),
            'HOST': config('DB_HOST', default='localhost'),
            'PORT': config('DB_PORT', default='3306'),
            'OPTIONS': {
                'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
                'charset': 'utf8mb4',
            },
        }
    }

# HTTPS settings - CRITIQUE pour la sécurité
SECURE_SSL_REDIRECT = True  # Force HTTPS pour toutes les connexions
SECURE_HSTS_SECONDS = 31536000  # 1 year - Strict Transport Security
SECURE_HSTS_INCLUDE_SUBDOMAINS = True  # Inclut les sous-domaines
SECURE_HSTS_PRELOAD = True  # Permet le preload HSTS

# SSL/TLS avancés
SECURE_REDIRECT_EXEMPT = []  # URLs exemptées de la redirection HTTPS
SECURE_SSL_HOST = None  # Host pour la redirection SSL
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')  # Pour les proxies/load balancers

# Cookie settings
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True

# Security headers
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Content Security Policy
CSP_DEFAULT_SRC = ("'self'",)
CSP_STYLE_SRC = ("'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdn.jsdelivr.net")
CSP_SCRIPT_SRC = ("'self'", "'unsafe-inline'", "https://code.jquery.com", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com")
CSP_FONT_SRC = ("'self'", "https://fonts.gstatic.com", "https://cdn.jsdelivr.net")
CSP_IMG_SRC = ("'self'", "data:", "https:")
CSP_CONNECT_SRC = ("'self'", "https://api.example.com")  # Adjust for your APIs

# Additional security headers
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'
SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin'

# Cookie settings
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
CSRF_COOKIE_SAMESITE = 'Strict'

# Additional security settings
SECURE_REDIRECT_EXEMPT = []  # URLs that should be exempt from HTTPS redirect
SECURE_SSL_HOST = None  # Host to redirect to for SSL

# Admin security
ADMIN_URL = config('ADMIN_URL', default='admin/')  # Change default admin URL

# Static files - Configuration pour cloud et local
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# WhiteNoise pour servir les fichiers statiques sur cloud
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Configuration pour hébergeurs cloud (désactiver les logs fichiers si pas de filesystem)
try:
    # Logging pour déploiement local
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'file': {
                'level': 'ERROR',
                'class': 'logging.FileHandler',
                'filename': os.path.join(BASE_DIR, 'logs', 'django_error.log'),
            },
        },
        'loggers': {
            'django': {
                'handlers': ['file'],
                'level': 'ERROR',
                'propagate': True,
            },
        },
    }
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
except:
    # Logging simplifié pour cloud (pas de filesystem)
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'level': 'ERROR',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            'django': {
                'handlers': ['console'],
                'level': 'ERROR',
                'propagate': True,
            },
        },
    }