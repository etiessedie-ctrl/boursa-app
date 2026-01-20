"""
Services de préprocessing des données pour les prévisions financières
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Classe pour le préprocessing des données financières avant les prévisions
    """

    def __init__(self):
        self.scalers = {}
        self.imputers = {}

    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                            strategy: str = 'interpolate',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans les données

        Args:
            df: DataFrame à traiter
            strategy: Stratégie ('interpolate', 'forward_fill', 'backward_fill', 'mean', 'median')
            columns: Colonnes spécifiques à traiter (toutes si None)

        Returns:
            DataFrame avec valeurs manquantes traitées
        """
        try:
            df_processed = df.copy()

            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in columns:
                if col not in df_processed.columns:
                    continue

                if strategy == 'interpolate':
                    df_processed[col] = df_processed[col].interpolate(method='linear')
                elif strategy == 'forward_fill':
                    df_processed[col] = df_processed[col].fillna(method='ffill')
                elif strategy == 'backward_fill':
                    df_processed[col] = df_processed[col].fillna(method='bfill')
                elif strategy == 'mean':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                elif strategy == 'median':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                elif strategy == 'zero':
                    df_processed[col] = df_processed[col].fillna(0)

            # Remplir les NaN restants avec 0
            df_processed = df_processed.fillna(0)

            return df_processed

        except Exception as e:
            raise Exception(f"Erreur lors du traitement des valeurs manquantes: {str(e)}")

    @staticmethod
    def detect_outliers(df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Détecte les outliers dans les données

        Args:
            df: DataFrame à analyser
            columns: Colonnes à analyser
            method: Méthode ('iqr', 'zscore', 'isolation_forest')
            threshold: Seuil pour la détection

        Returns:
            Dictionnaire avec les outliers détectés
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            outliers_info = {}

            for col in columns:
                if col not in df.columns:
                    continue

                data = df[col].dropna()

                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                elif method == 'zscore':
                    z_scores = np.abs((data - data.mean()) / data.std())
                    outliers = data[z_scores > threshold]
                else:
                    outliers = pd.Series(dtype=float)

                outliers_info[col] = {
                    'count': len(outliers),
                    'indices': outliers.index.tolist(),
                    'values': outliers.values.tolist(),
                    'percentage': len(outliers) / len(data) * 100 if len(data) > 0 else 0
                }

            return outliers_info

        except Exception as e:
            raise Exception(f"Erreur lors de la détection d'outliers: {str(e)}")

    @staticmethod
    def remove_outliers(df: pd.DataFrame,
                       outliers_info: Dict[str, Any],
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Supprime les outliers détectés

        Args:
            df: DataFrame original
            outliers_info: Informations sur les outliers
            columns: Colonnes à traiter

        Returns:
            DataFrame sans outliers
        """
        try:
            df_clean = df.copy()

            if columns is None:
                columns = list(outliers_info.keys())

            indices_to_remove = set()

            for col in columns:
                if col in outliers_info:
                    indices_to_remove.update(outliers_info[col]['indices'])

            if indices_to_remove:
                df_clean = df_clean.drop(list(indices_to_remove))

            return df_clean.reset_index(drop=True)

        except Exception as e:
            raise Exception(f"Erreur lors de la suppression d'outliers: {str(e)}")

    def normalize_data(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalise les données

        Args:
            df: DataFrame à normaliser
            columns: Colonnes à normaliser
            method: Méthode ('standard', 'minmax')

        Returns:
            Tuple (DataFrame normalisé, paramètres de normalisation)
        """
        try:
            df_normalized = df.copy()

            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            params = {}

            for col in columns:
                if col not in df.columns:
                    continue

                data = df[col].values.reshape(-1, 1)

                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    continue

                scaled_data = scaler.fit_transform(data)
                df_normalized[col] = scaled_data.flatten()

                # Sauvegarder les paramètres
                params[col] = {
                    'scaler': scaler,
                    'method': method
                }

            return df_normalized, params

        except Exception as e:
            raise Exception(f"Erreur lors de la normalisation: {str(e)}")

    def inverse_normalize(self, df_normalized: pd.DataFrame,
                         normalization_params: Dict[str, Any],
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse la normalisation

        Args:
            df_normalized: DataFrame normalisé
            normalization_params: Paramètres de normalisation
            columns: Colonnes à traiter

        Returns:
            DataFrame avec valeurs originales
        """
        try:
            df_original = df_normalized.copy()

            if columns is None:
                columns = list(normalization_params.keys())

            for col in columns:
                if col in normalization_params and col in df_original.columns:
                    scaler = normalization_params[col]['scaler']
                    data = df_original[col].values.reshape(-1, 1)
                    original_data = scaler.inverse_transform(data)
                    df_original[col] = original_data.flatten()

            return df_original

        except Exception as e:
            raise Exception(f"Erreur lors de l'inverse normalisation: {str(e)}")

    @staticmethod
    def create_technical_indicators(df: pd.DataFrame,
                                   price_col: str = 'Close',
                                   include_volume: bool = True) -> pd.DataFrame:
        """
        Crée des indicateurs techniques pour les données financières

        Args:
            df: DataFrame avec données OHLC
            price_col: Colonne de prix à utiliser
            include_volume: Inclure les indicateurs de volume

        Returns:
            DataFrame avec indicateurs techniques
        """
        try:
            df_indicators = df.copy()

            # Moyennes mobiles
            df_indicators['SMA_20'] = df_indicators[price_col].rolling(window=20).mean()
            df_indicators['SMA_50'] = df_indicators[price_col].rolling(window=50).mean()
            df_indicators['EMA_20'] = df_indicators[price_col].ewm(span=20).mean()

            # RSI
            delta = df_indicators[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df_indicators[price_col].ewm(span=12).mean()
            ema_26 = df_indicators[price_col].ewm(span=26).mean()
            df_indicators['MACD'] = ema_12 - ema_26
            df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9).mean()

            # Bollinger Bands
            sma_20 = df_indicators[price_col].rolling(window=20).mean()
            std_20 = df_indicators[price_col].rolling(window=20).std()
            df_indicators['BB_Upper'] = sma_20 + (std_20 * 2)
            df_indicators['BB_Lower'] = sma_20 - (std_20 * 2)

            # Volume indicators (si Volume existe)
            if include_volume and 'Volume' in df_indicators.columns:
                df_indicators['Volume_SMA_20'] = df_indicators['Volume'].rolling(window=20).mean()

            # Remplir les NaN avec interpolation
            df_indicators = df_indicators.interpolate(method='linear')
            df_indicators = df_indicators.fillna(0)

            return df_indicators

        except Exception as e:
            raise Exception(f"Erreur lors de la création d'indicateurs techniques: {str(e)}")

    @staticmethod
    def prepare_time_series(df: pd.DataFrame,
                           date_col: Optional[str] = None,
                           target_col: str = 'Close',
                           freq: Optional[str] = None) -> pd.DataFrame:
        """
        Prépare les données pour l'analyse de séries temporelles

        Args:
            df: DataFrame à préparer
            date_col: Colonne de date (auto-détectée si None)
            target_col: Colonne cible
            freq: Fréquence ('D', 'H', 'W', etc.) - auto-détectée si None

        Returns:
            DataFrame préparé pour les séries temporelles
        """
        try:
            df_ts = df.copy()

            # Détecter la colonne de date si non spécifiée
            if date_col is None:
                # Recherche plus intelligente des colonnes de date
                potential_date_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    # Colonnes contenant des mots-clés de date
                    if any(keyword in col_lower for keyword in ['date', 'time', 'datetime', 'timestamp']):
                        potential_date_cols.append((col, 3))  # Haute priorité
                    # Colonnes avec des types datetime
                    elif df[col].dtype in ['datetime64[ns]', 'datetime64[ns, UTC]']:
                        potential_date_cols.append((col, 2))  # Moyenne priorité
                    # Colonnes avec des valeurs qui ressemblent à des dates
                    elif df[col].dtype == 'object':
                        sample_values = df[col].dropna().head(5)
                        if len(sample_values) > 0:
                            try:
                                pd.to_datetime(sample_values, errors='coerce')
                                potential_date_cols.append((col, 1))  # Basse priorité
                            except:
                                pass

                # Trier par priorité et prendre la meilleure
                if potential_date_cols:
                    potential_date_cols.sort(key=lambda x: x[1], reverse=True)
                    date_col = potential_date_cols[0][0]

            # Si aucune colonne de date trouvée, créer un index temporel
            if date_col is None or date_col not in df_ts.columns:
                # Essayer d'inférer la fréquence à partir des données existantes
                if freq is None:
                    # Par défaut, supposer des données quotidiennes
                    freq = 'D'
                    # Essayer de détecter une fréquence basée sur le nombre de données
                    # Par exemple, si beaucoup de données, pourrait être horaire
                    if len(df) > 1000:
                        freq = 'H'  # Potentiellement des données horaires
                    elif len(df) > 100:
                        freq = 'D'  # Données quotidiennes
                    else:
                        freq = 'W'  # Données hebdomadaires si peu de points

                df_ts['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq=freq)
                date_col = 'Date'

            # Convertir en datetime si nécessaire
            if date_col in df_ts.columns:
                # Essayer différents formats de date
                date_formats = [
                    None,  # Auto-détection
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%m/%d/%Y',
                    '%Y/%m/%d',
                    '%d-%m-%Y',
                    '%Y%m%d',
                    '%d.%m.%Y'
                ]

                converted = False
                for fmt in date_formats:
                    try:
                        if fmt is None:
                            df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                        else:
                            df_ts[date_col] = pd.to_datetime(df_ts[date_col], format=fmt, errors='coerce')

                        # Vérifier que la conversion a réussi
                        if not df_ts[date_col].isnull().all():
                            converted = True
                            break
                    except:
                        continue

                if not converted:
                    raise ValueError(f"Impossible de convertir la colonne '{date_col}' en dates")

                df_ts = df_ts.set_index(date_col)

            # Trier par date
            df_ts = df_ts.sort_index()

            # Inférer et appliquer la fréquence appropriée
            if freq is None:
                try:
                    inferred_freq = pd.infer_freq(df_ts.index)
                    if inferred_freq is not None:
                        freq = inferred_freq
                    else:
                        freq = 'D'  # Défaut
                except:
                    freq = 'D'  # Défaut en cas d'erreur

            # S'assurer que l'index est continu avec la fréquence appropriée
            try:
                df_ts = df_ts.asfreq(freq, method='pad')
            except:
                # Si asfreq échoue, essayer avec une méthode plus simple
                pass

            # Vérifier que la colonne cible existe
            if target_col not in df_ts.columns:
                raise ValueError(f"Colonne cible '{target_col}' non trouvée")

            return df_ts

        except Exception as e:
            raise Exception(f"Erreur lors de la préparation des séries temporelles: {str(e)}")

    @staticmethod
    def split_train_test(df: pd.DataFrame,
                        target_col: str,
                        train_ratio: float = 0.8,
                        time_series: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divise les données en ensembles d'entraînement et de test

        Args:
            df: DataFrame à diviser
            target_col: Colonne cible
            train_ratio: Ratio pour l'entraînement
            time_series: Respecter l'ordre temporel

        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        try:
            if time_series:
                # Pour les séries temporelles, garder l'ordre chronologique
                split_idx = int(len(df) * train_ratio)
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
            else:
                # Division aléatoire
                train_df = df.sample(frac=train_ratio, random_state=42)
                test_df = df.drop(train_df.index)

            # Séparer features et target
            feature_cols = [col for col in df.columns if col != target_col]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise Exception(f"Erreur lors de la division train/test: {str(e)}")
