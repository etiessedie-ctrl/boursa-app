from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from core.constants import AVAILABLE_MODELS
from core.models import UserLocation
import pandas as pd
import numpy as np
import os
from django.conf import settings
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io

def home(request):
    return render(request, 'core/accueil.html')

def description(request):
    return render(request, 'core/description.html')

def tests(request):
    # Récupérer les informations de fichier depuis la session
    context = {
        'file_columns': request.session.get('file_columns', []),
        'current_file': request.session.get('uploaded_file', ''),
        'file_dtypes': request.session.get('file_dtypes', {}),
        'file_rows': request.session.get('file_rows', 0),
        'last_test_result': request.session.get('last_test_result'),
        'last_test_type': request.session.get('last_test_type'),
        'last_selected_columns': request.session.get('last_selected_columns', []),
    }
    return render(request, 'core/tests.html', context)

def handle_forecast_request(request):
    """Handle forecast request (AJAX or regular POST)"""
    try:
        # Get form data
        selected_model = request.POST.get('selected_model')
        target_column = request.POST.get('target_column')
        forecast_type = request.POST.get('forecast_type')
        forecast_interval = request.POST.get('forecast_interval')
        forecast_steps = int(request.POST.get('forecast_steps', 5))
        filename = request.POST.get('filename')

        # Validate required data
        if not all([selected_model, target_column, filename]):
            error_msg = 'Données manquantes pour la prévision'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg})
            else:
                return render_forecast_error(request, error_msg)

        # Get file path from session avec validation robuste
        file_path = request.session.get('file_path')
        if not file_path:
            error_msg = 'Session expirée. Veuillez recharger votre fichier de données depuis la page d\'accueil.'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg, 'session_expired': True})
            else:
                from django.contrib import messages
                messages.error(request, error_msg)
                from django.shortcuts import redirect
                return redirect('core:accueil')

        if not os.path.exists(file_path):
            error_msg = 'Fichier de données introuvable. Il a peut-être été supprimé ou déplacé. Veuillez recharger votre fichier.'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg, 'file_missing': True})
            else:
                from django.contrib import messages
                messages.error(request, error_msg)
                from django.shortcuts import redirect
                return redirect('core:accueil')

        # Load data avec gestion d'erreur améliorée
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            error_msg = f'Erreur lors de la lecture du fichier: {str(e)}. Le fichier est peut-être corrompu.'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg, 'file_corrupted': True})
            else:
                from django.contrib import messages
                messages.error(request, error_msg)
                from django.shortcuts import redirect
                return redirect('core:accueil')

        if target_column not in df.columns:
            error_msg = f'Colonne cible "{target_column}" non trouvée dans les données. Les colonnes disponibles sont: {", ".join(df.columns.tolist())}'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg, 'column_missing': True})
            else:
                return render_forecast_error(request, error_msg)
        
        # Basic forecast simulation (placeholder)
        # In a real implementation, this would call the appropriate ML model
        target_data = df[target_column].dropna()
        if len(target_data) < 10:
            error_msg = 'Pas assez de données pour effectuer une prévision (minimum 10 points requis)'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg})
            else:
                return render_forecast_error(request, error_msg)
        
        # Utiliser le vrai service de machine learning
        from core.services.machine_learning import ForecastingService
        from core.services.preprocessing import DataPreprocessor
        
        # Préparer les données pour les séries temporelles de manière robuste
        try:
            # Utiliser DataPreprocessor pour gérer intelligemment l'index temporel
            df_prepared = DataPreprocessor.prepare_time_series(
                df=df,
                date_col=None,  # Auto-détection de la colonne de date
                target_col=target_column,
                freq=None  # Auto-détection de la fréquence
            )
            
            # Extraire la série cible préparée
            target_data = df_prepared[target_column].dropna()
            
            # Vérifier à nouveau que nous avons assez de données après préparation
            if len(target_data) < 10:
                error_msg = 'Pas assez de données valides pour effectuer une prévision après préparation (minimum 10 points requis)'
                if request.POST.get('ajax') == 'true':
                    return JsonResponse({'success': False, 'message': error_msg})
                else:
                    return render_forecast_error(request, error_msg)
            
            # Valider la fréquence temporelle pour les modèles qui en ont besoin
            if isinstance(target_data.index, pd.DatetimeIndex):
                # Détecter la fréquence réelle des données
                detected_freq = pd.infer_freq(target_data.index)
                if detected_freq is None:
                    # Si la fréquence ne peut pas être inférée, rééchantillonner
                    target_data = target_data.asfreq('D', method='pad')
                    detected_freq = 'D'
                
                # Avertir si la fréquence semble irrégulière pour certains modèles
                if selected_model.upper() in ['ARIMA', 'SARIMA', 'ARMA'] and detected_freq not in ['D', 'H', 'W', 'M']:
                    # Pour ces modèles, une fréquence régulière est critique
                    target_data = target_data.asfreq('D', method='pad')
            
        except Exception as e:
            error_msg = f'Erreur lors de la préparation des données temporelles: {str(e)}'
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg})
            else:
                return render_forecast_error(request, error_msg)
        
        forecast_result = ForecastingService.forecast(
            data=target_data,
            model_name=selected_model,
            steps=forecast_steps,
            config={}  # Utiliser la config par défaut
        )
        
        if not forecast_result['success']:
            error_msg = forecast_result.get('message', 'Erreur lors de la prévision')
            if request.POST.get('ajax') == 'true':
                return JsonResponse({'success': False, 'message': error_msg})
            else:
                return render_forecast_error(request, error_msg)
        
        forecast_values = forecast_result['predictions']
        
        # Sauvegarder dans l'historique des prévisions
        try:
            from core.models import ForecastHistory
            import time
            start_time = time.time()  # This should be set at the beginning of the function
            
            ForecastHistory.objects.create(
                model_name=selected_model,
                target_column=target_column,
                filename=filename,
                forecast_steps=forecast_steps,
                forecast_type=forecast_type,
                forecast_interval=forecast_interval,
                predictions=forecast_values,
                metrics=forecast_result.get('metrics', {}),
                execution_time=time.time() - start_time if 'start_time' in locals() else None,
                success=True
            )
        except Exception as e:
            # Log error but don't fail the forecast
            print(f"Error saving forecast history: {e}")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Période': range(1, forecast_steps + 1),
            'Valeur_Prévue': forecast_values
        })
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(target_data)), target_data.values, label='Données Historiques', color='blue')
        plt.plot(range(len(target_data), len(target_data) + forecast_steps), forecast_values, 
                label='Prévisions', color='red', linestyle='--')
        plt.title(f'Prévisions {selected_model.upper()} - {target_column}')
        plt.xlabel('Période')
        plt.ylabel(target_column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Create results table
        results_html = forecast_df.to_html(
            classes='min-w-full divide-y divide-gray-200',
            index=False,
            border=0,
            justify='left'
        )
        
        if request.POST.get('ajax') == 'true':
            # Return JSON for AJAX requests
            return JsonResponse({
                'success': True,
                'forecast_plot': plot_base64,
                'forecast_results': results_html,
                'model': selected_model,
                'target_column': target_column,
                'steps': forecast_steps
            })
        else:
            # Return template render for regular POST requests
            context = {
                'file_columns': request.session.get('file_columns', []),
                'current_file': request.session.get('uploaded_file', ''),
                'file_dtypes': request.session.get('file_dtypes', {}),
                'file_rows': request.session.get('file_rows', 0),
                'available_models': AVAILABLE_MODELS,
                'forecast_results': results_html,
                'forecast_plot': plot_base64,
                'selected_model': selected_model,
                'target_column': target_column,
                'forecast_steps': forecast_steps,
                'bootstrap_json': {
                    'columns': request.session.get('file_columns', []),
                    'filename': request.session.get('uploaded_file', ''),
                    'dtypes': request.session.get('file_dtypes', {}),
                    'is_post_request': True
                }
            }
            return render(request, 'core/previsions.html', context)

    except Exception as e:
        error_msg = f'Erreur lors de la prévision: {str(e)}'
        if request.POST.get('ajax') == 'true':
            return JsonResponse({'success': False, 'message': error_msg})
        else:
            return render_forecast_error(request, error_msg)


def render_forecast_error(request, error_message):
    """Render forecast page with error message"""
    context = {
        'file_columns': request.session.get('file_columns', []),
        'current_file': request.session.get('uploaded_file', ''),
        'file_dtypes': request.session.get('file_dtypes', {}),
        'file_rows': request.session.get('file_rows', 0),
        'available_models': AVAILABLE_MODELS,
        'forecast_results': f'<div class="alert alert-danger p-4 bg-red-100 text-red-700 rounded-lg">{error_message}</div>',
        'bootstrap_json': {
            'columns': request.session.get('file_columns', []),
            'filename': request.session.get('uploaded_file', ''),
            'dtypes': request.session.get('file_dtypes', {}),
            'is_post_request': True
        }
    }
    return render(request, 'core/previsions.html', context)

def previsions(request):
    """View for ML forecasting page"""
    if request.method == 'POST':
        if request.POST.get('ajax') == 'true':
            # Handle AJAX forecast request
            return handle_forecast_request(request)
        else:
            # Handle regular POST request for forecast
            return handle_forecast_request(request)

    # GET request - show the form
    # Vérifier si nous avons des données en session
    file_columns = request.session.get('file_columns', [])
    current_file = request.session.get('uploaded_file', '')
    file_path = request.session.get('file_path')

    # Contrairement à avant, on affiche toujours la page previsions
    # Si on n'a pas de données, on commence à l'étape 1 (import)
    # Si on a des données, on va directement à l'étape 2 (configuration)

    # Préparer le contexte de base
    context = {
        'file_columns': file_columns,
        'current_file': current_file,
        'file_dtypes': request.session.get('file_dtypes', {}),
        'file_rows': request.session.get('file_rows', 0),
        'available_models': AVAILABLE_MODELS,
        'bootstrap_json': {
            'columns': file_columns,
            'filename': current_file,
            'dtypes': request.session.get('file_dtypes', {}),
            'is_post_request': request.method == 'POST'
        }
    }

    # Si nous avons des données valides en session, vérifier leur cohérence
    if file_columns and current_file and file_path and os.path.exists(file_path):
        try:
            if current_file.endswith('.csv'):
                df_check = pd.read_csv(file_path)
            else:
                df_check = pd.read_excel(file_path)

            # Vérifier que les colonnes correspondent
            if set(file_columns) == set(df_check.columns.tolist()):
                # Données valides - on peut aller à l'étape 2
                context['bootstrap_json']['is_post_request'] = False  # Pour déclencher l'étape 2
                return render(request, 'core/previsions.html', context)
            else:
                # Colonnes changées - nettoyer la session et recommencer
                from django.contrib import messages
                messages.warning(request, 'Les colonnes du fichier ont changé. Veuillez recharger votre fichier.')
                # Nettoyer la session
                for key in ['file_columns', 'uploaded_file', 'file_path', 'file_dtypes', 'file_rows']:
                    if key in request.session:
                        del request.session[key]
        except Exception as e:
            # Erreur de lecture - nettoyer la session et recommencer
            from django.contrib import messages
            messages.error(request, f'Erreur lors de la lecture du fichier: {str(e)}. Veuillez recharger votre fichier.')
            # Nettoyer la session
            for key in ['file_columns', 'uploaded_file', 'file_path', 'file_dtypes', 'file_rows']:
                if key in request.session:
                    del request.session[key]
    else:
        # Pas de données en session - nettoyer au cas où
        for key in ['file_columns', 'uploaded_file', 'file_path', 'file_dtypes', 'file_rows']:
            if key in request.session:
                del request.session[key]

    # Dans tous les cas, afficher la page previsions (commencera à l'étape 1)
    return render(request, 'core/previsions.html', context)

def historiques(request):
    """View for displaying both test and forecast history (combined page)"""
    from core.models import TestHistory, ForecastHistory

    # Tests history
    test_history = TestHistory.objects.all()[:50]
    total_tests = TestHistory.objects.count()
    significant_tests = TestHistory.objects.filter(is_significant=True).count()
    unique_test_files = TestHistory.objects.values('filename').distinct().count()

    # Forecasts history
    forecast_history = ForecastHistory.objects.all()[:50]
    total_forecasts = ForecastHistory.objects.count()
    last_forecast = ForecastHistory.objects.first()
    last_forecast_date = last_forecast.timestamp.strftime('%d/%m/%Y %H:%M') if last_forecast else None
    unique_forecast_files = ForecastHistory.objects.values('filename').distinct().count()

    # Define test categories for template
    moyenne_tests = ['student_t', 'paired_t', 'one_sample_t']
    variance_tests = ['f_test', 'levene', 'bartlett']
    non_param_tests = ['wilcoxon', 'mannwhitney', 'kruskal', 'friedman']
    temporel_tests = ['autocorrelation', 'stationarity']

    context = {
        # Tests
        'history': test_history,
        'total_tests': total_tests,
        'significant_tests': significant_tests,
        'unique_files': unique_test_files,
        'moyenne_tests': moyenne_tests,
        'variance_tests': variance_tests,
        'non_param_tests': non_param_tests,
        'temporel_tests': temporel_tests,
        # Forecasts
        'forecast_history': forecast_history,
        'total_forecasts': total_forecasts,
        'last_forecast_date': last_forecast_date,
        'unique_forecast_files': unique_forecast_files,
    }

    return render(request, 'core/historiques.html', context)

def visualisation(request):
    """View for data visualization page"""
    if request.method == 'POST' and request.POST.get('ajax') == 'true':
        # Handle AJAX request for custom chart generation
        return handle_custom_chart_request(request)

    # GET request - show the visualization page
    context = {
        'file_columns': request.session.get('file_columns', []),
        'current_file': request.session.get('uploaded_file', ''),
        'file_dtypes': request.session.get('file_dtypes', {}),
        'file_rows': request.session.get('file_rows', 0),
    }

    # Generate visualizations if we have data
    file_path = request.session.get('file_path')
    if file_path and os.path.exists(file_path):
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Generate all visualizations
            from core.services.plots import VisualizationService
            visualizations = VisualizationService.generate_all_visualizations(df)

            context.update({
                'visualizations': visualizations,
                'data_loaded': True,
                'data_shape': df.shape,
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            })

        except Exception as e:
            context.update({
                'error': f"Erreur lors de la génération des visualisations: {str(e)}",
                'data_loaded': False,
            })
    else:
        context.update({
            'data_loaded': False,
            'message': "Aucune donnée chargée. Veuillez d'abord uploader un fichier sur la page d'accueil.",
        })

    return render(request, 'core/visualisation.html', context)

def handle_custom_chart_request(request):
    """Handle AJAX request for custom chart generation"""
    try:
        # Get parameters from POST data
        x_col = request.POST.get('x_column')
        y_col = request.POST.get('y_column')
        chart_type = request.POST.get('chart_type')

        # Validate required parameters
        if not all([x_col, y_col, chart_type]):
            return JsonResponse({
                'success': False,
                'message': 'Paramètres manquants pour le graphique personnalisé'
            })

        # Get file path from session
        file_path = request.session.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return JsonResponse({
                'success': False,
                'message': 'Fichier de données non trouvé. Veuillez recharger vos données.'
            })

        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Generate the appropriate chart
        from core.services.plots import VisualizationService

        if chart_type == 'scatter':
            chart_data = VisualizationService.create_scatter_plot(
                df, x_col, y_col, f"Nuage de points: {x_col} vs {y_col}"
            )
        elif chart_type == 'line':
            chart_data = VisualizationService.create_line_plot(
                df, x_col, y_col, f"Graphique en ligne: {x_col} vs {y_col}"
            )
        elif chart_type == 'bar':
            chart_data = VisualizationService.create_bar_plot(
                df, x_col, y_col, f"Graphique en barres: {x_col} vs {y_col}"
            )
        else:
            return JsonResponse({
                'success': False,
                'message': f'Type de graphique non supporté: {chart_type}'
            })

        return JsonResponse({
            'success': True,
            'chart_data': chart_data,
            'chart_type': chart_type,
            'x_column': x_col,
            'y_column': y_col
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Erreur lors de la génération du graphique: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["GET", "POST"])
def location_api(request):
    """
    API endpoint for user location tracking
    GET: Retrieve all active user locations
    POST: Update/create user location
    """
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)

            device_id = data.get('username')  # Using username field from frontend
            latitude = data.get('lat')
            longitude = data.get('lon')
            accuracy = data.get('accuracy')

            if not all([device_id, latitude, longitude]):
                return JsonResponse({
                    'success': False,
                    'message': 'Données de localisation manquantes'
                }, status=400)

            # Validate coordinates
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                return JsonResponse({
                    'success': False,
                    'message': 'Coordonnées invalides'
                }, status=400)

            # Update or create location
            location, created = UserLocation.objects.update_or_create(
                device_id=device_id,
                defaults={
                    'latitude': latitude,
                    'longitude': longitude,
                    'accuracy': accuracy,
                    'timestamp': timezone.now(),
                    'last_seen': timezone.now(),
                    'is_active': True
                }
            )

            if not created:
                location.update_last_seen()

            return JsonResponse({
                'success': True,
                'message': 'Position mise à jour'
            })

        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'message': 'Données JSON invalides'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur serveur: {str(e)}'
            }, status=500)

    elif request.method == 'GET':
        try:
            # Get active locations (seen within last 10 minutes)
            locations = UserLocation.get_active_locations(max_age_minutes=10)

            users_data = []
            for location in locations:
                users_data.append({
                    'username': location.device_id,
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'accuracy': location.accuracy,
                    'timestamp': location.last_seen.isoformat(),
                    'active_users': 1  # For compatibility with frontend
                })

            return JsonResponse({
                'users': users_data,
                'total_active': len(users_data)
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur lors de la récupération des positions: {str(e)}'
            }, status=500)

def cartographie(request):
    # Clean up old inactive locations (older than 1 hour)
    from datetime import timedelta
    cutoff_time = timezone.now() - timedelta(hours=1)
    UserLocation.objects.filter(last_seen__lt=cutoff_time).update(is_active=False)
    
    # Get some statistics for the template
    active_locations = UserLocation.get_active_locations(max_age_minutes=10)
    total_active = active_locations.count()
    
    context = {
        'total_active_users': total_active,
        'last_update': timezone.now(),
    }
    return render(request, 'core/cartographie.html', context)

def resultats(request):
    """View for displaying test results"""
    # Get last test result from session
    last_result = request.session.get('last_test_result')
    last_test_type = request.session.get('last_test_type')
    last_selected_columns = request.session.get('last_selected_columns', [])
    
    if not last_result or 'error' in last_result:
        # No results available
        context = {
            'test_name': None,
        }
        return render(request, 'core/resultats.html', context)
    
    # Format results text
    results_text = ""
    if 'test_name' in last_result:
        results_text += f"Test: {last_result['test_name']}\n\n"
    
    if 'statistic' in last_result and 'p_value' in last_result:
        results_text += f"Statistique: {last_result['statistic']:.4f}\n"
        results_text += f"Valeur p: {last_result['p_value']:.4f}\n"
    
    if 'alpha' in last_result:
        results_text += f"Niveau alpha: {last_result['alpha']}\n"
    
    if 'normality_assessment' in last_result:
        results_text += f"Évaluation de normalité: {last_result['normality_assessment']}\n"
    
    if 'sample_size' in last_result:
        results_text += f"Taille d'échantillon: {last_result['sample_size']}\n"
    
    if 'estimated_mean' in last_result:
        results_text += f"Moyenne estimée: {last_result['estimated_mean']:.4f}\n"
    
    if 'estimated_std' in last_result:
        results_text += f"Écart-type estimé: {last_result['estimated_std']:.4f}\n"
    
    if 'description' in last_result:
        results_text += f"\nDescription: {last_result['description']}\n"
    
    # Get column name (first selected column for single-column tests)
    column = last_selected_columns[0] if last_selected_columns else "N/A"
    
    # Generate visualizations
    histogram = None
    qqplot = None
    timeseries = None
    
    file_path = request.session.get('file_path')
    if file_path and os.path.exists(file_path):
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            if column in df.columns:
                data_series = df[column]
                
                # Generate histogram
                from core.services.plots import VisualizationService
                histogram = VisualizationService.create_histogram(
                    data_series, f"Histogramme de {column}"
                )
                
                # Generate Q-Q plot for normality tests
                if last_test_type in ['kolmogorov_smirnov', 'shapiro_wilk']:
                    qqplot = VisualizationService.create_qqplot(
                        data_series, f"Q-Q Plot de {column}"
                    )
                
                # Generate time series if data looks temporal
                if len(df) > 10:
                    # Create a simple line plot of the data
                    plt.figure(figsize=(14, 7))
                    plt.plot(range(len(data_series)), data_series.values, marker='o', linestyle='-', alpha=0.7)
                    plt.title(f"Série temporelle: {column}")
                    plt.xlabel('Index')
                    plt.ylabel(column)
                    plt.grid(True, alpha=0.3)
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    timeseries = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    timeseries = f"data:image/png;base64,{timeseries}"
                    plt.close()
        
        except Exception as e:
            # Ignore visualization errors
            pass
    
    context = {
        'test_name': last_result.get('test_name', last_test_type),
        'column': column,
        'results_text': results_text,
        'histogram': histogram,
        'qqplot': qqplot,
        'timeseries': timeseries,
        'can_download': True,
    }
    
    return render(request, 'core/resultats.html', context)

def previsions_history(request):
    """View for displaying forecast history"""
    from core.models import ForecastHistory

    # Get all forecast history
    history = ForecastHistory.objects.all()[:50]  # Limit to last 50 forecasts
    total_forecasts = ForecastHistory.objects.count()

    # Get last forecast date
    last_forecast = ForecastHistory.objects.first()
    last_forecast_date = last_forecast.timestamp.strftime('%d/%m/%Y %H:%M') if last_forecast else None

    # Get unique files count
    unique_files_count = ForecastHistory.objects.values('filename').distinct().count()

    context = {
        'history': history,
        'total_forecasts': total_forecasts,
        'last_forecast_date': last_forecast_date,
        'unique_files_count': unique_files_count,
    }

    return render(request, 'core/previsions_history.html', context)

def historiques(request):
    """View for displaying both test and forecast history"""
    from core.models import TestHistory, ForecastHistory

    # Get test history
    test_history = TestHistory.objects.all()[:50]  # Limit to last 50 tests
    total_tests = TestHistory.objects.count()

    # Get last test date
    last_test = TestHistory.objects.first()
    last_test_date = last_test.timestamp.strftime('%d/%m/%Y %H:%M') if last_test else None

    # Get unique test files count
    unique_test_files_count = TestHistory.objects.values('filename').distinct().count()

    # Get forecast history
    forecast_history = ForecastHistory.objects.all()[:50]  # Limit to last 50 forecasts
    total_forecasts = ForecastHistory.objects.count()

    # Get last forecast date
    last_forecast = ForecastHistory.objects.first()
    last_forecast_date = last_forecast.timestamp.strftime('%d/%m/%Y %H:%M') if last_forecast else None

    # Get unique forecast files count
    unique_forecast_files_count = ForecastHistory.objects.values('filename').distinct().count()

    # Serialize test history for JavaScript
    import json
    test_history_json = json.dumps([
        {
            'id': test.id,
            'test_name': test.test_name,
            'test_type': test.test_type,
            'filename': test.filename,
            'columns_used': test.selected_columns if isinstance(test.selected_columns, list) else [],
            'timestamp': test.timestamp.strftime('%d/%m/%Y %H:%M'),
            'p_value': float(test.p_value) if test.p_value is not None else None,
            'stat_value': float(test.statistic) if test.statistic is not None else None,
            'full_results': test.result_data if isinstance(test.result_data, dict) else {},
            'interpretation': test.interpretation,
        }
        for test in test_history
    ])

    # Parse forecast_values for template display
    for forecast in forecast_history:
        try:
            forecast.parsed_predictions = json.loads(forecast.forecast_values) if forecast.forecast_values else []
        except:
            forecast.parsed_predictions = []

    context = {
        'test_history': test_history,
        'total_tests': total_tests,
        'last_test_date': last_test_date,
        'unique_test_files_count': unique_test_files_count,
        'forecast_history': forecast_history,
        'total_forecasts': total_forecasts,
        'last_forecast_date': last_forecast_date,
        'unique_forecast_files_count': unique_forecast_files_count,
        'test_history_json': test_history_json,
    }

    # Parse selected_columns for template display
    for test in test_history:
        try:
            test.parsed_columns = test.selected_columns if isinstance(test.selected_columns, list) else []
        except:
            test.parsed_columns = []

    return render(request, 'core/historiques.html', context)

def clear_all_history(request):
    if request.method == 'POST':
        try:
            from core.models import TestHistory
            # Delete all test history
            deleted_count = TestHistory.objects.all().delete()
            return JsonResponse({
                'success': True, 
                'message': f'Historique des tests supprimé avec succès. {deleted_count[0]} éléments supprimés.'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur lors de la suppression: {str(e)}'
            })
    return render(request, 'core/historique.html')

def download_last_result(request):
    """Download the last test result as JSON"""
    last_result = request.session.get('last_test_result')
    last_test_type = request.session.get('last_test_type')
    last_selected_columns = request.session.get('last_selected_columns', [])
    
    if not last_result:
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="no_results.txt"'
        response.write("Aucun résultat disponible")
        return response
    
    # Create JSON response
    import json
    result_data = {
        'test_type': last_test_type,
        'selected_columns': last_selected_columns,
        'timestamp': timezone.now().isoformat(),
        'results': last_result
    }
    
    response = HttpResponse(content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename="test_results.json"'
    response.write(json.dumps(result_data, indent=2, default=str))
    return response

def download_forecast_history(request):
    """Download forecast history as CSV"""
    try:
        from core.models import ForecastHistory
        import csv
        
        forecasts = ForecastHistory.objects.all()
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="forecast_history.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Date', 'Modèle', 'Colonne cible', 'Fichier', 'Étapes', 
            'Prédictions', 'Succès', 'Temps exécution'
        ])
        
        for forecast in forecasts:
            predictions_str = ','.join([str(p) for p in forecast.predictions]) if forecast.predictions else ''
            writer.writerow([
                forecast.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                forecast.model_name,
                forecast.target_column,
                forecast.filename or '',
                forecast.forecast_steps,
                predictions_str,
                'Oui' if forecast.success else 'Non',
                forecast.execution_time or ''
            ])
        
        return response
    except Exception as e:
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="error.txt"'
        response.write(f"Erreur lors de l'export: {str(e)}")
        return response

def clear_forecast_history(request):
    if request.method == 'POST':
        try:
            from core.models import ForecastHistory
            # Delete all forecast history
            deleted_count = ForecastHistory.objects.all().delete()
            return JsonResponse({
                'success': True, 
                'message': f'Historique des prévisions supprimé avec succès. {deleted_count[0]} éléments supprimés.'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur lors de la suppression: {str(e)}'
            })
    return JsonResponse({'success': False, 'message': 'Méthode non autorisée'})

def download_forecast(request, forecast_id):
    # TODO: Implement specific forecast download
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="forecast_{forecast_id}.csv"'
    response.write(f"forecast_id,date,value\n{forecast_id},2024-01-01,100.5\n")
    return response