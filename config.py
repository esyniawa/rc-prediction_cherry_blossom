scalers_config = {
    'countdown_to_first': {
        'column_name': 'countdown_to_first',
        'scaler_type': 'minmax',
        'scaler_kwargs': None
    },
    'countdown_to_full': {
        'column_name': 'countdown_to_full',
        'scaler_type': 'minmax',
        'scaler_kwargs': None
    },
    'temps_to_first': {
        'column_name': 'temps_to_first',
        'scaler_type': 'minmax',
        'scaler_kwargs': None
    },
    'temps_to_full': {
        'column_name': 'temps_to_full',
        'scaler_type': 'minmax',
        'scaler_kwargs': None
    },
    'lat': {
        'column_name': 'lat',
        'scaler_type': 'robust',
        'scaler_kwargs': {'quantile_range': (5, 95)}
    },
    'lng': {
        'column_name': 'lng',
        'scaler_type': 'robust',
        'scaler_kwargs': {'quantile_range': (5, 95)}
    }
}
