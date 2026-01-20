import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ParametricTests:
    """
    Classe pour effectuer des tests statistiques paramétriques et de normalité
    """

    @staticmethod
    def kolmogorov_smirnov_test(data: List[float]) -> Dict[str, Any]:
        """
        Test de Kolmogorov-Smirnov pour tester la normalité
        Compare la distribution des données à une distribution normale
        """
        try:
            # Nettoyer les données (enlever NaN et infinis)
            data_clean = [x for x in data if pd.notna(x) and np.isfinite(x)]

            if len(data_clean) < 3:
                return {'error': 'Au moins 3 valeurs sont requises pour le test'}

            # Calculer les paramètres de la distribution normale
            mean = np.mean(data_clean)
            std = np.std(data_clean, ddof=1)

            # Effectuer le test
            statistic, p_value = stats.kstest(data_clean, 'norm', args=(mean, std))

            # Interprétation
            alpha = 0.05
            normality = "Distribution non normale" if p_value < alpha else "Distribution compatible avec la normale"

            return {
                'test_name': 'Test de Kolmogorov-Smirnov',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'normality_assessment': normality,
                'sample_size': len(data_clean),
                'estimated_mean': float(mean),
                'estimated_std': float(std),
                'description': 'Test de conformité à une distribution normale'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Kolmogorov-Smirnov: {str(e)}'}

    @staticmethod
    def shapiro_wilk_test(data: List[float]) -> Dict[str, Any]:
        """
        Test de Shapiro-Wilk pour tester la normalité
        Plus puissant que K-S pour les petits échantillons
        """
        try:
            # Nettoyer les données
            data_clean = [x for x in data if pd.notna(x) and np.isfinite(x)]

            if len(data_clean) < 3:
                return {'error': 'Au moins 3 valeurs sont requises pour le test'}
            if len(data_clean) > 5000:
                return {'error': 'Le test de Shapiro-Wilk n\'est pas adapté aux grands échantillons (>5000)'}

            # Effectuer le test
            statistic, p_value = stats.shapiro(data_clean)

            # Interprétation
            alpha = 0.05
            normality = "Distribution non normale" if p_value < alpha else "Distribution compatible avec la normale"

            return {
                'test_name': 'Test de Shapiro-Wilk',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'normality_assessment': normality,
                'sample_size': len(data_clean),
                'description': 'Test de normalité plus puissant que K-S pour les petits échantillons'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Shapiro-Wilk: {str(e)}'}

    @staticmethod
    def test_normality(data: List[float], test_type: str = 'auto') -> Dict[str, Any]:
        """
        Fonction wrapper pour choisir automatiquement le test de normalité
        """
        if test_type == 'auto':
            # Choisir automatiquement selon la taille de l'échantillon
            if len(data) <= 5000:
                return ParametricTests.shapiro_wilk_test(data)
            else:
                return ParametricTests.kolmogorov_smirnov_test(data)
        elif test_type == 'shapiro':
            return ParametricTests.shapiro_wilk_test(data)
        elif test_type == 'ks':
            return ParametricTests.kolmogorov_smirnov_test(data)
        else:
            return {'error': 'Type de test non reconnu'}