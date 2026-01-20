import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonParametricTests:
    """
    Classe pour effectuer des tests statistiques non paramétriques
    """

    @staticmethod
    def wilcoxon_test(data1: List[float], data2: List[float]) -> Dict[str, Any]:
        """
        Test de Wilcoxon pour échantillons appariés
        Compare deux échantillons appariés (dépendants)
        """
        try:
            # Vérifier que les données ont la même longueur
            if len(data1) != len(data2):
                return {
                    'error': 'Les deux échantillons doivent avoir la même taille pour le test de Wilcoxon'
                }

            # Effectuer le test
            statistic, p_value = stats.wilcoxon(data1, data2)

            # Interprétation
            alpha = 0.05
            interpretation = "H0 rejetée : différence significative entre les groupes" if p_value < alpha else "H0 acceptée : pas de différence significative"

            return {
                'test_name': 'Test de Wilcoxon',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'interpretation': interpretation,
                'sample_size': len(data1),
                'description': 'Test non paramétrique pour comparer deux échantillons appariés'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Wilcoxon: {str(e)}'}

    @staticmethod
    def mann_whitney_test(data1: List[float], data2: List[float]) -> Dict[str, Any]:
        """
        Test de Mann-Whitney pour groupes indépendants
        Compare deux groupes indépendants
        """
        try:
            # Effectuer le test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # Interprétation
            alpha = 0.05
            interpretation = "H0 rejetée : différence significative entre les groupes" if p_value < alpha else "H0 acceptée : pas de différence significative"

            return {
                'test_name': 'Test de Mann-Whitney',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'interpretation': interpretation,
                'sample_sizes': [len(data1), len(data2)],
                'description': 'Test non paramétrique pour comparer deux groupes indépendants'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Mann-Whitney: {str(e)}'}

    @staticmethod
    def kruskal_wallis_test(*groups: List[List[float]]) -> Dict[str, Any]:
        """
        Test de Kruskal-Wallis pour K groupes indépendants
        """
        try:
            if len(groups) < 2:
                return {'error': 'Au moins 2 groupes sont requis'}

            # Effectuer le test
            statistic, p_value = stats.kruskal(*groups)

            # Interprétation
            alpha = 0.05
            interpretation = "H0 rejetée : au moins un groupe diffère des autres" if p_value < alpha else "H0 acceptée : pas de différence significative entre les groupes"

            return {
                'test_name': 'Test de Kruskal-Wallis',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'interpretation': interpretation,
                'num_groups': len(groups),
                'sample_sizes': [len(group) for group in groups],
                'description': 'Test non paramétrique pour comparer K groupes indépendants'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Kruskal-Wallis: {str(e)}'}

    @staticmethod
    def friedman_test(*groups: List[List[float]]) -> Dict[str, Any]:
        """
        Test de Friedman pour K échantillons appariés
        """
        try:
            if len(groups) < 2:
                return {'error': 'Au moins 2 groupes sont requis'}

            # Vérifier que tous les groupes ont la même taille
            group_sizes = [len(group) for group in groups]
            if len(set(group_sizes)) != 1:
                return {'error': 'Tous les groupes doivent avoir la même taille pour le test de Friedman'}

            # Effectuer le test
            statistic, p_value = stats.friedmanchisquare(*groups)

            # Interprétation
            alpha = 0.05
            interpretation = "H0 rejetée : au moins un traitement diffère des autres" if p_value < alpha else "H0 acceptée : pas de différence significative entre les traitements"

            return {
                'test_name': 'Test de Friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'alpha': alpha,
                'interpretation': interpretation,
                'num_groups': len(groups),
                'sample_size_per_group': group_sizes[0],
                'description': 'Test non paramétrique pour comparer K traitements appariés'
            }

        except Exception as e:
            return {'error': f'Erreur lors du test de Friedman: {str(e)}'}

    @staticmethod
    def spearman_correlation(data1: List[float], data2: List[float]) -> Dict[str, Any]:
        """
        Corrélation de Spearman entre deux variables
        """
        try:
            # Effectuer le test
            correlation, p_value = stats.spearmanr(data1, data2)

            # Interprétation de la force de corrélation
            abs_corr = abs(correlation)
            if abs_corr < 0.3:
                strength = "très faible"
            elif abs_corr < 0.5:
                strength = "faible"
            elif abs_corr < 0.7:
                strength = "modérée"
            elif abs_corr < 0.9:
                strength = "forte"
            else:
                strength = "très forte"

            # Interprétation de la significativité
            alpha = 0.05
            significance = "significative" if p_value < alpha else "non significative"

            return {
                'test_name': 'Corrélation de Spearman',
                'correlation_coefficient': float(correlation),
                'p_value': float(p_value),
                'alpha': alpha,
                'correlation_strength': strength,
                'significance': significance,
                'sample_size': len(data1),
                'description': 'Mesure de la corrélation de rang entre deux variables'
            }

        except Exception as e:
            return {'error': f'Erreur lors du calcul de la corrélation de Spearman: {str(e)}'}