"""
Services statistiques pour l'analyse de données financières
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalysis:
    """
    Classe pour l'analyse statistique des données financières
    """

    @staticmethod
    def descriptive_statistics(df: pd.DataFrame,
                             columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calcule les statistiques descriptives des données

        Args:
            df: DataFrame à analyser
            columns: Colonnes spécifiques à analyser

        Returns:
            Dictionnaire avec statistiques descriptives
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            stats_dict = {}

            for col in columns:
                if col not in df.columns:
                    continue

                data = df[col].dropna()

                if len(data) == 0:
                    continue

                stats_dict[col] = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'missing_values': df[col].isnull().sum(),
                    'missing_percentage': df[col].isnull().sum() / len(df) * 100
                }

            return stats_dict

        except Exception as e:
            raise Exception(f"Erreur lors du calcul des statistiques descriptives: {str(e)}")

    @staticmethod
    def stationarity_tests(data: pd.Series,
                          alpha: float = 0.05) -> Dict[str, Any]:
        """
        Effectue des tests de stationarité (ADF et KPSS)

        Args:
            data: Série temporelle à tester
            alpha: Niveau de signification

        Returns:
            Résultats des tests de stationarité
        """
        try:
            data_clean = data.dropna()

            if len(data_clean) < 10:
                return {'error': 'Au moins 10 observations requises pour les tests de stationarité'}

            results = {}

            # Test ADF (Augmented Dickey-Fuller)
            try:
                adf_result = adfuller(data_clean, autolag='AIC')
                results['adf'] = {
                    'test_name': 'Augmented Dickey-Fuller',
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {key: float(value) for key, value in adf_result[4].items()},
                    'stationary': adf_result[1] < alpha,
                    'lags_used': int(adf_result[2]),
                    'observations': int(adf_result[3])
                }
            except Exception as e:
                results['adf'] = {'error': f'Erreur test ADF: {str(e)}'}

            # Test KPSS
            try:
                kpss_result = kpss(data_clean, regression='c', nlags='auto')
                results['kpss'] = {
                    'test_name': 'KPSS',
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {key: float(value) for key, value in kpss_result[3].items()},
                    'stationary': kpss_result[1] >= alpha,
                    'lags_used': int(kpss_result[2])
                }
            except Exception as e:
                results['kpss'] = {'error': f'Erreur test KPSS: {str(e)}'}

            # Conclusion générale
            adf_stationary = results.get('adf', {}).get('stationary', False)
            kpss_stationary = results.get('kpss', {}).get('stationary', False)

            if adf_stationary and kpss_stationary:
                conclusion = "Série stationnaire"
            elif not adf_stationary and not kpss_stationary:
                conclusion = "Série non stationnaire"
            else:
                conclusion = "Résultats contradictoires - analyse complémentaire requise"

            results['conclusion'] = conclusion

            return results

        except Exception as e:
            return {'error': f'Erreur lors des tests de stationarité: {str(e)}'}

    @staticmethod
    def normality_tests(data: pd.Series,
                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        Effectue des tests de normalité

        Args:
            data: Données à tester
            alpha: Niveau de signification

        Returns:
            Résultats des tests de normalité
        """
        try:
            data_clean = data.dropna()

            if len(data_clean) < 3:
                return {'error': 'Au moins 3 observations requises pour les tests de normalité'}

            results = {}

            # Test de Shapiro-Wilk
            try:
                if len(data_clean) <= 5000:  # Shapiro limité à 5000 observations
                    statistic, p_value = stats.shapiro(data_clean)
                    results['shapiro'] = {
                        'test_name': 'Shapiro-Wilk',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'normal': p_value >= alpha
                    }
            except Exception as e:
                results['shapiro'] = {'error': f'Erreur test Shapiro: {str(e)}'}

            # Test de Kolmogorov-Smirnov
            try:
                mean = data_clean.mean()
                std = data_clean.std()
                statistic, p_value = stats.kstest(data_clean, 'norm', args=(mean, std))
                results['kolmogorov'] = {
                    'test_name': 'Kolmogorov-Smirnov',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'normal': p_value >= alpha
                }
            except Exception as e:
                results['kolmogorov'] = {'error': f'Erreur test Kolmogorov: {str(e)}'}

            # Test d'Anderson-Darling
            try:
                result = stats.anderson(data_clean, dist='norm')
                results['anderson'] = {
                    'test_name': 'Anderson-Darling',
                    'statistic': float(result.statistic),
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist(),
                    'normal': result.statistic < result.critical_values[2]  # 5% level
                }
            except Exception as e:
                results['anderson'] = {'error': f'Erreur test Anderson: {str(e)}'}

            # Statistiques descriptives pour l'interprétation
            skewness = float(data_clean.skew())
            kurtosis = float(data_clean.kurtosis())

            results['descriptive'] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jb_statistic': len(data_clean)/6 * (skewness**2 + (kurtosis**2)/4),
                'sample_size': len(data_clean)
            }

            # Conclusion générale
            normal_tests = [test for test in ['shapiro', 'kolmogorov', 'anderson']
                          if test in results and 'normal' in results[test]]

            if normal_tests:
                normal_count = sum(1 for test in normal_tests if results[test]['normal'])
                if normal_count >= len(normal_tests) * 0.6:  # Majorité des tests
                    conclusion = "Distribution compatible avec la normale"
                else:
                    conclusion = "Distribution non normale"
            else:
                conclusion = "Impossible de conclure"

            results['conclusion'] = conclusion

            return results

        except Exception as e:
            return {'error': f'Erreur lors des tests de normalité: {str(e)}'}

    @staticmethod
    def autocorrelation_analysis(data: pd.Series,
                                lags: Optional[int] = None,
                                alpha: float = 0.05) -> Dict[str, Any]:
        """
        Analyse l'autocorrélation des données

        Args:
            data: Série temporelle à analyser
            lags: Nombre de lags à analyser
            alpha: Niveau de signification

        Returns:
            Résultats de l'analyse d'autocorrélation
        """
        try:
            data_clean = data.dropna()

            if len(data_clean) < 10:
                return {'error': 'Au moins 10 observations requises pour l\'analyse d\'autocorrélation'}

            if lags is None:
                lags = min(20, len(data_clean) // 5)

            results = {}

            # Fonction d'autocorrélation (ACF)
            try:
                acf_values = []
                for lag in range(1, lags + 1):
                    corr = data_clean.corr(data_clean.shift(lag))
                    acf_values.append(float(corr) if not np.isnan(corr) else 0)

                results['acf'] = {
                    'values': acf_values,
                    'lags': list(range(1, len(acf_values) + 1)),
                    'significant_lags': [i+1 for i, val in enumerate(acf_values)
                                       if abs(val) > 1.96 / np.sqrt(len(data_clean))]
                }
            except Exception as e:
                results['acf'] = {'error': f'Erreur calcul ACF: {str(e)}'}

            # Test de Ljung-Box
            try:
                lb_test = acorr_ljungbox(data_clean, lags=lags, return_df=False)
                results['ljung_box'] = {
                    'test_name': 'Ljung-Box',
                    'statistics': lb_test[0].tolist(),
                    'p_values': lb_test[1].tolist(),
                    'lags': list(range(1, len(lb_test[0]) + 1)),
                    'autocorrelated': any(p < alpha for p in lb_test[1])
                }
            except Exception as e:
                results['ljung_box'] = {'error': f'Erreur test Ljung-Box: {str(e)}'}

            return results

        except Exception as e:
            return {'error': f'Erreur lors de l\'analyse d\'autocorrélation: {str(e)}'}

    @staticmethod
    def correlation_analysis(df: pd.DataFrame,
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Analyse les corrélations entre variables

        Args:
            df: DataFrame à analyser
            method: Méthode de corrélation ('pearson', 'spearman', 'kendall')

        Returns:
            Matrice de corrélation et analyses
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.shape[1] < 2:
                return {'error': 'Au moins 2 variables numériques requises'}

            # Matrice de corrélation
            corr_matrix = numeric_df.corr(method=method)

            # Identifier les paires fortement corrélées
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Seuil arbitraire
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'forte' if abs(corr_value) > 0.8 else 'modérée'
                        })

            results = {
                'correlation_matrix': corr_matrix.to_dict(),
                'method': method,
                'strong_correlations': strong_correlations,
                'variables': corr_matrix.columns.tolist()
            }

            return results

        except Exception as e:
            return {'error': f'Erreur lors de l\'analyse de corrélation: {str(e)}'}

    @staticmethod
    def volatility_analysis(data: pd.Series,
                          window: int = 20) -> Dict[str, Any]:
        """
        Analyse la volatilité des données financières

        Args:
            data: Série de prix/rendements
            window: Fenêtre pour le calcul de volatilité

        Returns:
            Mesures de volatilité
        """
        try:
            data_clean = data.dropna()

            if len(data_clean) < window:
                return {'error': f'Au moins {window} observations requises'}

            # Calcul des rendements si ce sont des prix
            if data_clean.min() > 0 and data_clean.max() / data_clean.min() > 2:
                # Probablement des prix, calculer les rendements
                returns = data_clean.pct_change().dropna()
            else:
                returns = data_clean

            results = {}

            # Volatilité historique (écart-type roulant)
            results['rolling_volatility'] = {
                'values': returns.rolling(window=window).std().dropna().tolist(),
                'window': window,
                'mean_volatility': float(returns.rolling(window=window).std().mean()),
                'max_volatility': float(returns.rolling(window=window).std().max()),
                'min_volatility': float(returns.rolling(window=window).std().min())
            }

            # Volatilité annualisée (approximation)
            daily_vol = returns.std()
            results['annualized_volatility'] = {
                'daily': float(daily_vol),
                'annualized': float(daily_vol * np.sqrt(252)),  # Approximation 252 jours de trading
                'monthly': float(daily_vol * np.sqrt(21))  # Approximation 21 jours de trading
            }

            # Value at Risk (VaR) simple
            confidence_levels = [0.95, 0.99]
            var_estimates = {}
            for conf in confidence_levels:
                var = np.percentile(returns, (1 - conf) * 100)
                var_estimates[f'var_{int(conf*100)}'] = float(var)

            results['value_at_risk'] = var_estimates

            return results

        except Exception as e:
            return {'error': f'Erreur lors de l\'analyse de volatilité: {str(e)}'}

    @staticmethod
    def seasonality_analysis(data: pd.Series,
                           freq: str = 'D') -> Dict[str, Any]:
        """
        Analyse la saisonnalité des données

        Args:
            data: Série temporelle
            freq: Fréquence des données

        Returns:
            Analyse de saisonnalité
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return {'error': 'Index temporel requis pour l\'analyse de saisonnalité'}

            data_clean = data.dropna()

            if len(data_clean) < 30:
                return {'error': 'Au moins 30 observations requises pour l\'analyse de saisonnalité'}

            results = {}

            # Analyse par jour de la semaine
            if freq in ['D', 'H']:
                data_clean['day_of_week'] = data_clean.index.dayofweek
                daily_means = data_clean.groupby('day_of_week').mean()

                results['day_of_week'] = {
                    'means': daily_means.to_dict(),
                    'days': ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                }

            # Analyse par mois
            data_clean['month'] = data_clean.index.month
            monthly_means = data_clean.groupby('month').mean()

            results['monthly'] = {
                'means': monthly_means.to_dict(),
                'months': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            }

            # Test de saisonnalité simple (différence entre max et min)
            if 'day_of_week' in results:
                day_variation = max(daily_means) - min(daily_means)
                results['day_seasonality'] = {
                    'variation': float(day_variation),
                    'significant': day_variation > data_clean.std() * 0.5
                }

            month_variation = max(monthly_means) - min(monthly_means)
            results['monthly_seasonality'] = {
                'variation': float(month_variation),
                'significant': month_variation > data_clean.std() * 0.5
            }

            return results

        except Exception as e:
            return {'error': f'Erreur lors de l\'analyse de saisonnalité: {str(e)}'}
