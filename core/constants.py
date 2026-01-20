# Available forecasting models
AVAILABLE_MODELS = {
    'arima': 'ARIMA (AutoRegressive Integrated Moving Average)',
    'sarima': 'SARIMA (Seasonal ARIMA)',
    'prophet': 'Prophet (Facebook)',
    'lstm': 'LSTM (Long Short-Term Memory)',
    'linear_regression': 'Régression Linéaire',
    # 'xgboost': 'XGBoost',  # TODO: Implement
    # 'random_forest': 'Random Forest',  # TODO: Implement
    # 'svm': 'Support Vector Machine (SVM)',  # TODO: Implement
    # 'exponential_smoothing': 'Lissage Exponentiel',  # TODO: Implement
    # 'holt_winters': 'Holt-Winters'  # TODO: Implement
}

# Model parameters defaults
MODEL_DEFAULT_PARAMS = {
    'arima': {'p': 1, 'd': 1, 'q': 1},
    'sarima': {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 's': 12},
    'prophet': {},
    'lstm': {'units': 50, 'epochs': 100, 'batch_size': 32},
    'linear_regression': {},
    # 'xgboost': {'n_estimators': 100, 'max_depth': 6},  # TODO: Implement
    # 'random_forest': {'n_estimators': 100, 'max_depth': None},  # TODO: Implement
    # 'svm': {'kernel': 'rbf', 'C': 1.0},  # TODO: Implement
    # 'exponential_smoothing': {'trend': 'add', 'seasonal': 'add'},  # TODO: Implement
    # 'holt_winters': {'trend': 'add', 'seasonal': 'add'}  # TODO: Implement
}