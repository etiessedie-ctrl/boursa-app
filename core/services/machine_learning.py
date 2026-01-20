"""
Services de Machine Learning pour les prévisions
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Classe de base pour tous les forecasters"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> None:
        """Entraîner le modèle"""
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Faire des prédictions"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Retourne le nom du modèle"""
        pass


class ARIMAForecaster(BaseForecaster):
    """Forecaster ARIMA"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.arima_class = ARIMA
        except ImportError:
            self.arima_class = None

    def fit(self, data: pd.Series, **kwargs) -> None:
        if self.arima_class is None:
            raise ImportError("statsmodels non installé")

        p = self.config.get('p', 1)
        d = self.config.get('d', 1)
        q = self.config.get('q', 1)

        self.model = self.arima_class(data, order=(p, d, q))
        self.fitted_model = self.model.fit()
        self.is_fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values

    def get_model_name(self) -> str:
        return "ARIMA"


class ProphetForecaster(BaseForecaster):
    """Forecaster Prophet"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            from prophet import Prophet
            self.prophet_class = Prophet
        except ImportError:
            self.prophet_class = None

    def fit(self, data: pd.Series, **kwargs) -> None:
        if self.prophet_class is None:
            raise ImportError("prophet non installé")

        # Préparer les données pour Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })

        self.model = self.prophet_class(**self.config)
        self.model.fit(df)
        self.is_fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(steps).values

    def get_model_name(self) -> str:
        return "Prophet"


class LSTMForecaster(BaseForecaster):
    """Forecaster LSTM basique"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
        except ImportError:
            self.tf = None

    def fit(self, data: pd.Series, **kwargs) -> None:
        if self.tf is None:
            raise ImportError("tensorflow non installé")

        # Configuration par défaut
        units = self.config.get('units', 50)
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)

        # Préparer les données
        data_values = data.values.reshape(-1, 1)

        # Créer séquences
        X, y = [], []
        sequence_length = 10
        for i in range(len(data_values) - sequence_length):
            X.append(data_values[i:i+sequence_length])
            y.append(data_values[i+sequence_length])

        X, y = np.array(X), np.array(y)

        # Construire le modèle
        self.model = self.Sequential([
            self.LSTM(units, input_shape=(sequence_length, 1)),
            self.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.last_sequence = data_values[-sequence_length:]
        self.is_fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(steps):
            pred = self.model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
            predictions.append(pred[0][0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0][0]

        return np.array(predictions)

    def get_model_name(self) -> str:
        return "LSTM"


class LinearRegressionForecaster(BaseForecaster):
    """Forecaster Régression Linéaire"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            from sklearn.linear_model import LinearRegression
            self.lr_class = LinearRegression
        except ImportError:
            self.lr_class = None

    def fit(self, data: pd.Series, **kwargs) -> None:
        if self.lr_class is None:
            raise ImportError("scikit-learn non installé")

        # Créer des features temporelles
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values

        self.model = self.lr_class()
        self.model.fit(X, y)
        self.last_index = len(data)
        self.is_fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modèle non entraîné")

        X_pred = np.arange(self.last_index, self.last_index + steps).reshape(-1, 1)
        return self.model.predict(X_pred)

    def get_model_name(self) -> str:
        return "Régression Linéaire"


class ForecastingService:
    """Service principal pour les prévisions"""

    MODEL_CLASSES = {
        'arima': ARIMAForecaster,
        'sarima': ARIMAForecaster,  # Même classe, config différente
        'prophet': ProphetForecaster,
        'lstm': LSTMForecaster,
        'linear_regression': LinearRegressionForecaster,
    }

    @staticmethod
    def get_available_models() -> List[str]:
        """Retourne la liste des modèles disponibles"""
        models = list(ForecastingService.MODEL_CLASSES.keys())

        # Ajouter les modèles pré-entraînés disponibles
        try:
            from core.services.pretrained_models import PretrainedModelService
            pretrained_models = PretrainedModelService.get_available_models()
            models.extend(pretrained_models)
        except ImportError:
            pass

        return models

    @staticmethod
    def create_forecaster(model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseForecaster:
        """Crée une instance de forecaster"""
        logger.debug(f"Création du forecaster pour le modèle: {model_name}")

        # Vérifier si c'est un modèle pré-entraîné
        try:
            from core.services.pretrained_models import PretrainedModelService
            logger.debug("Vérification si c'est un modèle pré-entraîné")
            if model_name in PretrainedModelService.get_available_models():
                logger.info(f"Modèle pré-entraîné détecté: {model_name}")
                service = PretrainedModelService()
                forecaster = service.load_model(model_name)
                logger.info(f"Forecaster pré-entraîné créé: {forecaster.get_model_name()}")
                return forecaster
        except ImportError as ie:
            logger.warning(f"Service de modèles pré-entraînés non disponible: {str(ie)}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle pré-entraîné {model_name}: {str(e)}")
            raise

        # Sinon, créer un forecaster classique
        logger.debug(f"Création d'un forecaster classique pour: {model_name}")
        if model_name not in ForecastingService.MODEL_CLASSES:
            available_models = list(ForecastingService.MODEL_CLASSES.keys())
            raise ValueError(f"Modèle {model_name} non supporté. Modèles disponibles: {available_models}")

        # Configuration spécifique par modèle
        from core.constants import MODEL_DEFAULT_PARAMS
        default_config = MODEL_DEFAULT_PARAMS.get(model_name, {})
        if config:
            default_config.update(config)

        logger.debug(f"Configuration du modèle {model_name}: {default_config}")
        forecaster = ForecastingService.MODEL_CLASSES[model_name](default_config)
        logger.debug(f"Forecaster classique créé: {forecaster.get_model_name()}")
        return forecaster

    @staticmethod
    def forecast(data: pd.Series, model_name: str, steps: int,
                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Effectue une prévision complète

        Args:
            data: Série temporelle
            model_name: Nom du modèle
            steps: Nombre de pas à prévoir
            config: Configuration du modèle

        Returns:
            Dictionnaire avec les résultats
        """
        logger.info(f"Début de la prévision - Modèle: {model_name}, Étapes: {steps}, Points de données: {len(data)}")

        try:
            # Étape 1: Validation des données d'entrée
            logger.debug(f"Validation des données d'entrée - Type: {type(data)}, Longueur: {len(data)}")
            if not isinstance(data, pd.Series):
                raise ValueError(f"Les données doivent être une pd.Series, reçu: {type(data)}")

            if len(data) == 0:
                raise ValueError("Les données ne peuvent pas être vides")

            if steps <= 0:
                raise ValueError(f"Le nombre d'étapes doit être positif, reçu: {steps}")

            logger.debug(f"Données validées - Index: {data.index[:3].tolist()}..., Valeurs: {data.values[:3].tolist()}...")

            # Étape 2: Création du forecaster
            logger.info(f"Création du forecaster pour le modèle: {model_name}")
            forecaster = ForecastingService.create_forecaster(model_name, config)
            logger.info(f"Forecaster créé avec succès: {forecaster.get_model_name()}")

            # Étape 3: Entraînement du modèle
            logger.info(f"Entraînement du modèle {model_name} avec {len(data)} points de données")
            forecaster.fit(data)
            logger.info(f"Modèle {model_name} entraîné avec succès")

            # Étape 4: Génération des prédictions
            logger.info(f"Génération de {steps} prédictions avec le modèle {model_name}")
            predictions = forecaster.predict(steps)
            logger.info(f"Prédictions générées avec succès - Shape: {predictions.shape}, Type: {type(predictions)}")

            # Étape 5: Validation des prédictions
            if predictions is None:
                raise ValueError("Les prédictions sont None")

            if len(predictions) != steps:
                logger.warning(f"Nombre de prédictions ({len(predictions)}) différent du nombre demandé ({steps})")

            # Convertir en liste pour la sérialisation JSON
            predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
            logger.debug(f"Prédictions converties en liste - Longueur: {len(predictions_list)}")

            result = {
                'success': True,
                'model': model_name,
                'predictions': predictions_list,
                'steps': steps,
                'message': f'Prévision {model_name} terminée avec succès'
            }

            logger.info(f"Prévision {model_name} terminée avec succès - {steps} prédictions générées")
            return result

        except Exception as e:
            logger.error(f"Erreur lors de la prévision {model_name}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'model': model_name,
                'error': str(e),
                'message': f'Erreur lors de la prévision {model_name}: {str(e)}'
            }