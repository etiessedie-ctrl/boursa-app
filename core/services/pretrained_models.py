"""
Service pour gérer les modèles de machine learning pré-entraînés
"""
import os
import joblib
from typing import Dict, Any, Optional, List
from django.conf import settings
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PretrainedForecaster:
    """Wrapper pour les modèles pré-entraînés qui implémente l'interface BaseForecaster"""

    def __init__(self, model_name: str, model: Any):
        self.model_name = model_name
        self.model = model
        self.is_fitted = True  # Les modèles pré-entraînés sont déjà entraînés

    def fit(self, data: pd.Series, **kwargs) -> None:
        """Les modèles pré-entraînés n'ont pas besoin d'être ré-entraînés"""
        logger.info(f"Modèle pré-entraîné {self.model_name} déjà prêt pour les prédictions")
        pass

    def predict(self, steps: int) -> np.ndarray:
        """Fait une prédiction pour le nombre d'étapes demandé"""
        # Pour les modèles de régression, nous avons besoin de features temporelles
        # Créer des features basées sur le temps (simplifié)
        # En pratique, il faudrait des features plus sophistiquées

        # Pour une démonstration, créer des features simples basées sur l'index
        # Dans un vrai scénario, il faudrait des features comme les prix passés, indicateurs techniques, etc.
        X_pred = np.arange(steps).reshape(-1, 1)

        try:
            predictions = self.model.predict(X_pred)
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec {self.model_name}: {str(e)}")
            # Retourner des valeurs par défaut en cas d'erreur
            return np.zeros(steps)

    def get_model_name(self) -> str:
        return f"Pretrained-{self.model_name}"


class PretrainedModelService:
    """Service pour charger et utiliser les modèles pré-entraînés"""

    # Mapping des noms de modèles vers les fichiers
    MODEL_FILES = {
        'random_forest': 'model_final_bourse_random_forest.joblib',
        'xgboost': 'model_final_bourse_xgboost.joblib',
    }

    def __init__(self):
        self.models_path = os.path.join(settings.BASE_DIR, 'core', 'ml_models')
        self.loaded_models = {}

    def get_model_path(self, model_name: str) -> str:
        """Retourne le chemin complet du fichier modèle"""
        if model_name not in self.MODEL_FILES:
            raise ValueError(f"Modèle {model_name} non trouvé dans MODEL_FILES")

        filename = self.MODEL_FILES[model_name]
        return os.path.join(self.models_path, filename)

    def load_model(self, model_name: str) -> PretrainedForecaster:
        """Charge un modèle et retourne un PretrainedForecaster"""
        if model_name in self.loaded_models:
            logger.info(f"Modèle {model_name} déjà chargé en mémoire")
            return self.loaded_models[model_name]

        try:
            model_path = self.get_model_path(model_name)
            logger.info(f"Tentative de chargement du modèle {model_name} depuis {model_path}")

            # Vérifier que le fichier existe
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fichier modèle non trouvé: {model_path}")

            # Vérifier que le fichier n'est pas vide
            if os.path.getsize(model_path) == 0:
                raise ValueError(f"Fichier modèle vide: {model_path}")

            logger.debug(f"Chargement du fichier joblib: {model_path}")

            # Tenter de charger le modèle
            try:
                model = joblib.load(model_path)
                logger.debug(f"Modèle {model_name} désérialisé avec succès")
            except Exception as load_error:
                logger.error(f"Erreur de désérialisation joblib pour {model_name}: {str(load_error)}")
                # Vérifier si c'est un problème de version scikit-learn
                try:
                    import sklearn
                    logger.info(f"Version scikit-learn actuelle: {sklearn.__version__}")
                except ImportError:
                    logger.warning("scikit-learn non installé")

                # Essayer de diagnostiquer le problème
                if "module" in str(load_error).lower() and "not found" in str(load_error).lower():
                    raise RuntimeError(f"Modèle {model_name} incompatible avec la version actuelle de scikit-learn. "
                                     f"Erreur: {str(load_error)}. "
                                     f"Recommandation: Ré-entraîner le modèle avec la version actuelle.")
                else:
                    raise RuntimeError(f"Fichier modèle corrompu ou incompatible: {model_path}. "
                                     f"Erreur: {str(load_error)}")

            # Vérifier que le modèle chargé est valide
            if model is None:
                raise ValueError(f"Modèle chargé est None: {model_path}")

            # Si le modèle est un dictionnaire (format spécial), extraire le vrai modèle
            if isinstance(model, dict):
                logger.debug(f"Modèle chargé est un dictionnaire, extraction du modèle réel")
                if 'model' not in model:
                    raise ValueError(f"Le dictionnaire modèle ne contient pas de clé 'model': {model_path}")
                actual_model = model['model']
                # Stocker les métadonnées pour utilisation future
                self.model_metadata = {
                    'feature_columns': model.get('feature_columns', []),
                    'threshold': model.get('threshold', 0.5)
                }
                logger.debug(f"Métadonnées extraites: {self.model_metadata}")
                model = actual_model

            # Vérifier que le modèle a les méthodes attendues
            if not hasattr(model, 'predict'):
                raise ValueError(f"Modèle chargé n'a pas de méthode predict: {model_path}")

            logger.debug(f"Validation du modèle {model_name} réussie")

            forecaster = PretrainedForecaster(model_name, model)
            self.loaded_models[model_name] = forecaster
            logger.info(f"Modèle {model_name} chargé et validé avec succès")
            return forecaster

        except Exception as e:
            logger.error(f"Échec du chargement du modèle {model_name}: {str(e)}")
            # Nettoyer le cache si le chargement a partiellement réussi
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            raise

    def predict(self, model_name: str, X) -> Any:
        """Fait une prédiction avec un modèle chargé"""
        forecaster = self.load_model(model_name)
        # Pour la compatibilité, permettre la prédiction directe avec X
        return forecaster.model.predict(X)

    @staticmethod
    def get_available_models() -> List[str]:
        """Retourne la liste des noms des modèles disponibles"""
        return list(PretrainedModelService.MODEL_FILES.keys())

    def unload_model(self, model_name: str) -> None:
        """Décharge un modèle de la mémoire"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Modèle {model_name} déchargé")

    def unload_all_models(self) -> None:
        """Décharge tous les modèles chargés"""
        self.loaded_models.clear()
        logger.info("Tous les modèles déchargés")


# Instance globale du service
pretrained_model_service = PretrainedModelService()