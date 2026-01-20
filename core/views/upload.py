from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import os
from django.conf import settings
import json
import glob
from datetime import datetime, timedelta

def cleanup_old_files():
    """Clean up files older than 1 hour in the datasets directory"""
    try:
        datasets_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        if not os.path.exists(datasets_dir):
            return

        # Get all files in datasets directory
        for file_path in glob.glob(os.path.join(datasets_dir, '*')):
            if os.path.isfile(file_path):
                # Check if file is older than 1 hour
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_age > timedelta(hours=1):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass  # Ignore errors when deleting old files
    except Exception:
        pass  # Ignore cleanup errors

def upload_file(request):
    # Clean up old files on each upload
    cleanup_old_files()

    if request.method == 'POST' and request.FILES.get('data_file'):
        uploaded_file = request.FILES['data_file']

        # Validate file extension
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'message': 'Format de fichier non supporté. Utilisez CSV ou Excel (.csv, .xlsx, .xls)'
            }, status=400)

        # Create unique filename to avoid conflicts
        import uuid
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', unique_filename)

        # Ensure directory exists and is writable
        datasets_dir = os.path.dirname(file_path)
        try:
            os.makedirs(datasets_dir, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(datasets_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur d\'accès au répertoire de stockage: {str(e)}'
            }, status=500)

        try:
            # Save file
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Read and validate the file
            try:
                if file_extension == '.csv':
                    # Try different encodings for CSV files
                    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    df = None
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(file_path, nrows=1000, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is None:
                        raise UnicodeDecodeError("Could not decode CSV file with any encoding")
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, nrows=1000)

                # Basic data validation
                if df is None or df.empty:
                    os.remove(file_path)  # Clean up invalid file
                    return JsonResponse({
                        'success': False,
                        'message': 'Le fichier est vide ou ne contient pas de données valides'
                    }, status=400)

                # Additional validation: check if we have at least some numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    os.remove(file_path)
                    return JsonResponse({
                        'success': False,
                        'message': 'Le fichier ne contient pas de colonnes numériques. Au moins une colonne numérique est requise pour les prévisions.'
                    }, status=400)

                # Validate column names
                invalid_cols = []
                for col in df.columns:
                    if not isinstance(col, str) or not col.strip():
                        invalid_cols.append(str(col))
                
                if invalid_cols:
                    os.remove(file_path)
                    return JsonResponse({
                        'success': False,
                        'message': f'Colonnes invalides détectées: {", ".join(invalid_cols)}. Les noms de colonnes ne peuvent pas être vides.'
                    }, status=400)

                # Get data types
                dtypes = {}
                for col in df.columns:
                    dtype_str = str(df[col].dtype)
                    # Convert pandas dtypes to more user-friendly names
                    if 'int' in dtype_str:
                        dtypes[col] = 'entier'
                    elif 'float' in dtype_str:
                        dtypes[col] = 'décimal'
                    elif 'datetime' in dtype_str or 'date' in dtype_str:
                        dtypes[col] = 'date'
                    elif 'bool' in dtype_str:
                        dtypes[col] = 'booléen'
                    else:
                        dtypes[col] = 'texte'

                # Get data preview (first 10 rows)
                preview_data = []
                for _, row in df.head(10).iterrows():
                    preview_data.append([str(val) for val in row.values])

                # Store file info in session
                request.session['uploaded_file'] = uploaded_file.name
                request.session['file_path'] = file_path
                request.session['file_columns'] = list(df.columns)
                request.session['file_dtypes'] = dtypes
                request.session['file_rows'] = len(df)

                return JsonResponse({
                    'success': True,
                    'filename': uploaded_file.name,
                    'columns': list(df.columns),
                    'dtypes': dtypes,
                    'rows': len(df),
                    'preview': preview_data,
                    'message': f'Fichier "{uploaded_file.name}" chargé avec succès. {len(df)} lignes et {len(df.columns)} colonnes détectées.'
                })

            except pd.errors.EmptyDataError:
                os.remove(file_path)
                return JsonResponse({
                    'success': False,
                    'message': 'Le fichier est vide ou ne contient pas de données valides'
                }, status=400)

            except pd.errors.ParserError:
                os.remove(file_path)
                return JsonResponse({
                    'success': False,
                    'message': 'Erreur de parsing du fichier. Vérifiez que le format est correct (CSV avec séparateur virgule ou Excel valide).'
                }, status=400)

            except UnicodeDecodeError:
                os.remove(file_path)
                return JsonResponse({
                    'success': False,
                    'message': 'Erreur d\'encodage du fichier. Essayez de sauvegarder le fichier CSV en UTF-8.'
                }, status=400)

            except Exception as e:
                os.remove(file_path)
                return JsonResponse({
                    'success': False,
                    'message': f'Erreur lors de la lecture du fichier: {str(e)}'
                }, status=400)

        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Erreur lors de la sauvegarde du fichier: {str(e)}'
            }, status=400)

    return JsonResponse({
        'success': False,
        'message': 'Aucun fichier uploadé ou méthode non autorisée'
    }, status=400)

def api_fetch(request):
    """
    Vue pour récupérer des données depuis des APIs boursières
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Méthode non autorisée'})

    try:
        import json
        data = json.loads(request.body)

        source = data.get('source') or ''
        symbol = data.get('symbol') or ''
        interval = data.get('interval', '1d')  # Default to '1d' if not provided
        api_key = data.get('api_key') or ''

        # Strip whitespace if values are strings
        if isinstance(source, str):
            source = source.strip()
        if isinstance(symbol, str):
            symbol = symbol.strip()
        if isinstance(api_key, str):
            api_key = api_key.strip()

        if not source or not symbol:
            return JsonResponse({
                'success': False,
                'message': 'Source et symbole sont requis.'
            })

        df = None

        if source == 'yahoo':
            try:
                import yfinance as yf

                # Télécharger les données (derniers 2 ans par défaut)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='2y', interval=interval)

                if df.empty:
                    return JsonResponse({
                        'success': False,
                        'message': f'Aucune donnée trouvée pour le symbole {symbol} sur Yahoo Finance.'
                    })

                # Reset index pour avoir Date comme colonne
                df = df.reset_index()
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            except ImportError:
                return JsonResponse({
                    'success': False,
                    'message': 'Yahoo Finance n\'est pas disponible. Installez yfinance.'
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Erreur Yahoo Finance: {str(e)}'
                })

        elif source == 'alpha_vantage':
            if not api_key:
                return JsonResponse({
                    'success': False,
                    'message': 'Clé API requise pour Alpha Vantage.'
                })

            try:
                import requests

                # API Alpha Vantage
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
                response = requests.get(url)
                data = response.json()

                if 'Error Message' in data:
                    return JsonResponse({
                        'success': False,
                        'message': f'Erreur Alpha Vantage: {data["Error Message"]}'
                    })

                if 'Time Series (Daily)' not in data:
                    return JsonResponse({
                        'success': False,
                        'message': 'Données non disponibles pour ce symbole.'
                    })

                # Convertir en DataFrame
                time_series = data['Time Series (Daily)']
                records = []
                for date, values in time_series.items():
                    record = {'Date': date}
                    for key, value in values.items():
                        # Enlever le préfixe "1. " ou "2. "
                        clean_key = key.split('. ', 1)[-1] if '. ' in key else key
                        # Convertir en float si la valeur n'est pas None
                        if value is not None and value != '':
                            try:
                                record[clean_key] = float(value)
                            except (ValueError, TypeError):
                                record[clean_key] = None
                        else:
                            record[clean_key] = None
                    records.append(record)

                df = pd.DataFrame(records)
                df = df.sort_values('Date')

            except ImportError:
                return JsonResponse({
                    'success': False,
                    'message': 'Requests n\'est pas disponible.'
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Erreur Alpha Vantage: {str(e)}'
                })

        elif source == 'iex_cloud':
            if not api_key:
                return JsonResponse({
                    'success': False,
                    'message': 'Clé API requise pour IEX Cloud.'
                })

            try:
                import requests

                # API IEX Cloud - données historiques
                url = f'https://cloud.iexapis.com/stable/stock/{symbol}/chart/2y?token={api_key}'
                response = requests.get(url)

                if response.status_code != 200:
                    return JsonResponse({
                        'success': False,
                        'message': f'Erreur IEX Cloud: {response.text}'
                    })

                data = response.json()
                if not data:
                    return JsonResponse({
                        'success': False,
                        'message': 'Aucune donnée disponible pour ce symbole.'
                    })

                # Convertir en DataFrame
                df = pd.DataFrame(data)
                # Nettoyer et convertir les dates
                df = df.dropna(subset=['date'])  # Supprimer les lignes sans date
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                df = df.dropna(subset=['date'])  # Supprimer les lignes avec dates invalides
                df = df.rename(columns={'date': 'Date'})

            except ImportError:
                return JsonResponse({
                    'success': False,
                    'message': 'Requests n\'est pas disponible.'
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Erreur IEX Cloud: {str(e)}'
                })

        elif source == 'polygon':
            if not api_key:
                return JsonResponse({
                    'success': False,
                    'message': 'Clé API requise pour Polygon.io.'
                })

            try:
                import requests

                # API Polygon.io - données historiques (2 ans maximum pour le plan gratuit)
                url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2022-01-01/2024-01-01?apiKey={api_key}'
                response = requests.get(url)

                if response.status_code != 200:
                    return JsonResponse({
                        'success': False,
                        'message': f'Erreur Polygon.io: {response.text}'
                    })

                data = response.json()
                if 'results' not in data or not data['results']:
                    return JsonResponse({
                        'success': False,
                        'message': 'Aucune donnée disponible pour ce symbole.'
                    })

                # Convertir en DataFrame
                records = []
                for result in data['results']:
                    # Convertir timestamp en date
                    from datetime import datetime
                    date = datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d')
                    record = {
                        'Date': date,
                        'Open': result.get('o'),
                        'High': result.get('h'),
                        'Low': result.get('l'),
                        'Close': result.get('c'),
                        'Volume': result.get('v')
                    }
                    records.append(record)

                df = pd.DataFrame(records)

            except ImportError:
                return JsonResponse({
                    'success': False,
                    'message': 'Requests n\'est pas disponible.'
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Erreur Polygon.io: {str(e)}'
                })

        else:
            return JsonResponse({
                'success': False,
                'message': f'Source "{source}" non supportée.'
            })

        if df is None or df.empty:
            return JsonResponse({
                'success': False,
                'message': 'Aucune donnée récupérée.'
            })

        # Sauvegarder temporairement les données
        import uuid
        filename = f"api_{source}_{symbol}_{uuid.uuid4()}.csv"
        temp_path = os.path.join(settings.MEDIA_ROOT, 'datasets', filename)

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        df.to_csv(temp_path, index=False)

        # Préparer la réponse
        dtypes = {}
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if 'int' in dtype_str:
                dtypes[col] = 'entier'
            elif 'float' in dtype_str:
                dtypes[col] = 'décimal'
            elif 'datetime' in dtype_str or 'date' in dtype_str:
                dtypes[col] = 'date'
            else:
                dtypes[col] = 'texte'

        # Aperçu des données
        preview_data = []
        for _, row in df.head(10).iterrows():
            preview_data.append([str(val) for val in row.values])

        # Stocker en session
        request.session['uploaded_file'] = f"{source.upper()}: {symbol}"
        request.session['file_path'] = temp_path
        request.session['file_columns'] = list(df.columns)
        request.session['file_dtypes'] = dtypes
        request.session['file_rows'] = len(df)

        return JsonResponse({
            'success': True,
            'filename': f"{source.upper()}: {symbol}",
            'columns': list(df.columns),
            'dtypes': dtypes,
            'rows': len(df),
            'preview': preview_data,
            'message': f'Données récupérées depuis {source.upper()} pour {symbol}. {len(df)} lignes chargées.'
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Données JSON invalides.'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Erreur inattendue: {str(e)}'
        })

def run_test(request):
    """
    Vue pour exécuter un test statistique sur les données uploadées
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Méthode non autorisée'})

    try:
        # Récupérer les données de la session
        df = get_uploaded_dataframe(request)
        if df is None:
            return JsonResponse({
                'success': False,
                'message': 'Aucune donnée chargée. Veuillez d\'abord importer un fichier.'
            })

        # Récupérer les paramètres du test depuis JSON ou POST
        if request.content_type == 'application/json':
            import json
            try:
                data = json.loads(request.body)
                test_type = data.get('test_type')
                selected_columns = data.get('columns', [])
            except json.JSONDecodeError:
                return JsonResponse({
                    'success': False,
                    'message': 'Données JSON invalides.'
                })
        else:
            test_type = request.POST.get('test_type')
            selected_columns = request.POST.getlist('columns[]')

        if not test_type:
            return JsonResponse({
                'success': False,
                'message': 'Veuillez sélectionner un type de test.'
            })

        if not selected_columns:
            return JsonResponse({
                'success': False,
                'message': 'Veuillez sélectionner au moins une colonne.'
            })

        # Importer les classes de tests
        from ..services.tests_non_parametric import NonParametricTests
        from ..services.tests_parametric import ParametricTests

        # Préparer les données selon le test
        result = None

        if test_type == 'wilcoxon':
            if len(selected_columns) != 2:
                return JsonResponse({
                    'success': False,
                    'message': 'Le test de Wilcoxon nécessite exactement 2 colonnes.'
                })

            col1, col2 = selected_columns
            if col1 not in df.columns or col2 not in df.columns:
                return JsonResponse({
                    'success': False,
                    'message': 'Colonnes sélectionnées non trouvées dans les données.'
                })

            # Nettoyer les données
            data1 = df[col1].dropna().tolist()
            data2 = df[col2].dropna().tolist()

            if len(data1) == 0 or len(data2) == 0:
                return JsonResponse({
                    'success': False,
                    'message': 'Une des colonnes sélectionnées est vide.'
                })

            result = NonParametricTests.wilcoxon_test(data1, data2)

        elif test_type == 'mannwhitney':
            if len(selected_columns) != 2:
                return JsonResponse({
                    'success': False,
                    'message': 'Le test de Mann-Whitney nécessite exactement 2 colonnes.'
                })

            col1, col2 = selected_columns
            if col1 not in df.columns or col2 not in df.columns:
                return JsonResponse({
                    'success': False,
                    'message': 'Colonnes sélectionnées non trouvées dans les données.'
                })

            data1 = df[col1].dropna().tolist()
            data2 = df[col2].dropna().tolist()

            if len(data1) == 0 or len(data2) == 0:
                return JsonResponse({
                    'success': False,
                    'message': 'Une des colonnes sélectionnées est vide.'
                })

            result = NonParametricTests.mann_whitney_test(data1, data2)

        elif test_type == 'kruskal':
            if len(selected_columns) < 2:
                return JsonResponse({
                    'success': False,
                    'message': 'Le test de Kruskal-Wallis nécessite au moins 2 colonnes.'
                })

            # Vérifier que toutes les colonnes existent
            for col in selected_columns:
                if col not in df.columns:
                    return JsonResponse({
                        'success': False,
                        'message': f'Colonne "{col}" non trouvée dans les données.'
                    })

            # Préparer les groupes
            groups = []
            for col in selected_columns:
                data = df[col].dropna().tolist()
                if len(data) == 0:
                    return JsonResponse({
                        'success': False,
                        'message': f'La colonne "{col}" est vide.'
                    })
                groups.append(data)

            result = NonParametricTests.kruskal_wallis_test(*groups)

        elif test_type == 'friedman':
            if len(selected_columns) < 2:
                return JsonResponse({
                    'success': False,
                    'message': 'Le test de Friedman nécessite au moins 2 colonnes.'
                })

            # Vérifier que toutes les colonnes existent
            for col in selected_columns:
                if col not in df.columns:
                    return JsonResponse({
                        'success': False,
                        'message': f'Colonne "{col}" non trouvée dans les données.'
                    })

            # Préparer les groupes
            groups = []
            for col in selected_columns:
                data = df[col].dropna().tolist()
                if len(data) == 0:
                    return JsonResponse({
                        'success': False,
                        'message': f'La colonne "{col}" est vide.'
                    })
                groups.append(data)

            result = NonParametricTests.friedman_test(*groups)

        elif test_type == 'spearman':
            if len(selected_columns) != 2:
                return JsonResponse({
                    'success': False,
                    'message': 'La corrélation de Spearman nécessite exactement 2 colonnes.'
                })

            col1, col2 = selected_columns
            if col1 not in df.columns or col2 not in df.columns:
                return JsonResponse({
                    'success': False,
                    'message': 'Colonnes sélectionnées non trouvées dans les données.'
                })

            data1 = df[col1].dropna().tolist()
            data2 = df[col2].dropna().tolist()

            if len(data1) == 0 or len(data2) == 0:
                return JsonResponse({
                    'success': False,
                    'message': 'Une des colonnes sélectionnées est vide.'
                })

            # Vérifier que les deux colonnes ont la même longueur après nettoyage
            min_len = min(len(data1), len(data2))
            data1 = data1[:min_len]
            data2 = data2[:min_len]

            result = NonParametricTests.spearman_correlation(data1, data2)

        elif test_type in ['kolmogorov_smirnov', 'shapiro_wilk']:
            if len(selected_columns) != 1:
                return JsonResponse({
                    'success': False,
                    'message': 'Les tests de normalité nécessitent exactement 1 colonne.'
                })

            col = selected_columns[0]
            if col not in df.columns:
                return JsonResponse({
                    'success': False,
                    'message': 'Colonne sélectionnée non trouvée dans les données.'
                })

            data = df[col].dropna().tolist()
            if len(data) == 0:
                return JsonResponse({
                    'success': False,
                    'message': 'La colonne sélectionnée est vide.'
                })

            if test_type == 'kolmogorov_smirnov':
                result = ParametricTests.kolmogorov_smirnov_test(data)
            else:  # shapiro_wilk
                result = ParametricTests.shapiro_wilk_test(data)

        else:
            return JsonResponse({
                'success': False,
                'message': f'Type de test "{test_type}" non reconnu.'
            })

        # Vérifier s'il y a eu une erreur
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'message': result['error']
            })

        # Stocker le résultat en session pour l'affichage
        request.session['last_test_result'] = result
        request.session['last_test_type'] = test_type
        request.session['last_selected_columns'] = selected_columns

        # Sauvegarder dans l'historique
        try:
            from core.models import TestHistory
            import time
            start_time = time.time()  # This should be set at the beginning of the function

            TestHistory.objects.create(
                test_type=test_type,
                test_name=result.get('test_name', test_type),
                selected_columns=selected_columns,
                filename=request.session.get('uploaded_file'),
                statistic=result.get('statistic'),
                p_value=result.get('p_value'),
                alpha=result.get('alpha', 0.05),
                sample_size=result.get('sample_size'),
                result_data=result,
                execution_time=time.time() - start_time if 'start_time' in locals() else None
            )
        except Exception as e:
            # Log error but don't fail the test execution
            print(f"Error saving test history: {e}")

        return JsonResponse({
            'success': True,
            'result': result,
            'message': f'Test "{result.get("test_name", test_type)}" exécuté avec succès.'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Erreur inattendue lors de l\'exécution du test: {str(e)}'
        })

def clear_session_file(request):
    # Clear uploaded file from session
    if 'uploaded_file' in request.session:
        del request.session['uploaded_file']
    if 'file_path' in request.session:
        del request.session['file_path']
    if 'file_columns' in request.session:
        del request.session['file_columns']
    if 'file_dtypes' in request.session:
        del request.session['file_dtypes']
    if 'file_rows' in request.session:
        del request.session['file_rows']
    return JsonResponse({'success': True})

def get_uploaded_dataframe(request):
    """Utility function to get the uploaded dataframe from session"""
    if 'file_path' in request.session and os.path.exists(request.session['file_path']):
        file_path = request.session['file_path']
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
        except Exception:
            pass
    return None