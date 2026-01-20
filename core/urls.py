from django.urls import path
from .views.dashboard import (
    home, description, tests, previsions, historiques, visualisation,
    cartographie, resultats, clear_all_history, 
    download_last_result, download_forecast_history, clear_forecast_history, download_forecast,
    location_api, handle_forecast_request
)
from .views.upload import upload_file, api_fetch, run_test, clear_session_file

app_name = 'core'

urlpatterns = [
    path('', home, name='accueil'),
    path('description/', description, name='description'),
    path('tests/', tests, name='tests'),
    path('previsions/', previsions, name='previsions'),
    path('historiques/', historiques, name='historiques'),
    path('visualisation/', visualisation, name='visualisation'),
    path('cartographie/', cartographie, name='cartographie'),
    path('resultats/', resultats, name='resultats'),
    path('clear-all-history/', clear_all_history, name='clear_all_history'),
    path('download-last-result/', download_last_result, name='download_last_result'),
    path('upload-file/', upload_file, name='upload_file'),
    path('api-fetch/', api_fetch, name='api_fetch'),
    path('run-test/', run_test, name='run_test'),
    path('clear-session-file/', clear_session_file, name='clear_session_file'),
    path('download-forecast-history/', download_forecast_history, name='download_forecast_history'),
    path('clear-forecast-history/', clear_forecast_history, name='clear_forecast_history'),
    path('download-forecast/<int:forecast_id>/', download_forecast, name='download_forecast'),
    path('locations/', location_api, name='location_api'),
    path('handle-forecast-request/', handle_forecast_request, name='handle_forecast_request'),
]