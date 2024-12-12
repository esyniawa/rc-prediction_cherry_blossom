# Here, the necessary datasets for the sakura blossom date prediciton are processed.
# load_sakura_data function reads the data, pre-processes them and returns features and labels.
# See comments below for more information.
import os.path
import pickle

# Datasets are downloaded from Kaggle:
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-cherry-blossom-data,
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-temperature-data
# https://www.kaggle.com/datasets/juanmah/world-cities

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from typing import Union, Type, Optional, Literal

from config import scalers_config


def parse_date(date_str: str, separator: str = ':'):
    day, month = map(int, date_str.split(separator))
    return day, month


def scale_column(
        df: pd.DataFrame,
        column_name: str,
        scaler_type: Union[str, Type] = "minmax",
        feature_range: tuple = (-1, 1),
        scaler_kwargs: Optional[dict] = None
) -> tuple[pd.DataFrame, object]:
    """
    Function to scale a column containing either scalar values or lists using various scikit-learn scalers.

    :param df: DataFrame containing the column to scale
    :type df: pd.DataFrame
    :param column_name: Name of the column to scale
    :type column_name: str
    :param scaler_type: Type of scaler to use. Can be either a string ('minmax', 'standard',
        'robust', 'maxabs') or a scikit-learn scaler class
    :type scaler_type: Union[str, Type]
    :param feature_range: Range for scaling (only used for MinMaxScaler)
    :type feature_range: tuple
    :param scaler_kwargs: Additional keyword arguments to pass to the scaler constructor
    :type scaler_kwargs: Optional[dict]
    :return: Tuple containing (Scaled DataFrame, fitted scaler)
    :rtype: tuple[pd.DataFrame, object]

    :example:

    Using MinMaxScaler with custom range:
    >>> df, scaler = scale_column(df, 'column_x', 'minmax', feature_range=(-2, 2))

    Using StandardScaler:
    >>> df, scaler = scale_column(df, 'column_x', 'standard')

    Using custom scaler with specific parameters:
    >>> df, scaler = scale_column(df, 'column_x', RobustScaler,
    ...     scaler_kwargs={'quantile_range': (10, 90)})
    """
    # Initialize scaler kwargs if None
    scaler_kwargs = scaler_kwargs or {}

    # Define scaler mapping
    SCALER_MAP = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler,
        'maxabs': MaxAbsScaler
    }

    # Get the appropriate scaler class
    if isinstance(scaler_type, str):
        scaler_type = scaler_type.lower()
        if scaler_type not in SCALER_MAP:
            raise ValueError(f"Unknown scaler type: {scaler_type}. "
                             f"Available types: {list(SCALER_MAP.keys())}")
        scaler_class = SCALER_MAP[scaler_type]
    else:
        scaler_class = scaler_type

    # Configure scaler with appropriate parameters
    if scaler_class == MinMaxScaler:
        scaler_kwargs['feature_range'] = feature_range

    # Create scaler instance
    scaler = scaler_class(**scaler_kwargs)

    # Check if the column contains lists/arrays
    is_list_column = isinstance(df[column_name].iloc[0], (list, np.ndarray))

    if is_list_column:
        # Concatenate all arrays in the column
        all_values = np.concatenate([np.array(x).flatten()
                                     for x in df[column_name] if len(x) > 0])

        # Fit scaler on all values
        scaler.fit(all_values.reshape(-1, 1))

        # Function to scale lists/arrays
        def scale_list(x):
            if len(x) == 0:
                return x
            return scaler.transform(np.array(x).reshape(-1, 1)).flatten().tolist()

        # Apply scaling to each list in the column
        df[column_name] = df[column_name].apply(scale_list)

    else:
        # Scale scalar values
        df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1, 1))

    return df, scaler


def interpolate_values(data_row: pd.Series,
                       min_points: int) -> list:
    # Replace empty values with NaN
    data_row = data_row.replace(['', 'missing', None], np.nan)

    # Check the number of valid points
    valid_points = data_row.dropna().size

    # Case 1: Insufficient data points
    if valid_points < min_points:
        return []

    # Case 2: Sufficient data points
    else:
        try:
            interpolated = data_row.interpolate(method='spline', order=2, limit_direction='both')
            return interpolated.tolist()
        except Exception:
            # Fallback to linear interpolation if spline fails
            interpolated = data_row.interpolate(method='linear', limit_direction='both')
            return interpolated.tolist()


def process_data(temp_df: pd.DataFrame,
                 first_bloom_df: pd.DataFrame,
                 full_bloom_df: pd.DataFrame,
                 cities_df: pd.DataFrame,
                 start_month: int = 8,
                 start_day: int = 1,
                 drop_inadequate_temps: bool = True) -> pd.DataFrame:
    """
    Function to merge the different datasets for sakura prediction

    :param temp_df: Dataframe with temperature data of different japanese cities
    :param first_bloom_df: Dataframe with the beginning of the sakura bloom dates for different japanese cities
    :param full_bloom_df: Dataframe with the date of the full sakura bloom for different japanese cities
    :param cities_df: Dataframe with geographic information about japanese cities
    :param start_month: Month of when the temperature data begins
    :param start_day: Day of when the temperature data begins
    :return: Merged dataframe
    """
    assert 1 <= start_month <= 12, "Start month must be between 1 and 12"
    assert 1 <= start_day <= 31, "Start day must be between 1 and 31"

    print("Processing data:")
    # Convert cities data to focus on Japanese cities
    print("Get japanese cities...")
    japan_cities = cities_df[cities_df['country'] == 'Japan'].copy()
    japan_cities = japan_cities[['city_ascii', 'lat', 'lng']]

    # Process bloom dates
    def extract_years_and_dates(bloom_df):
        # Get numeric columns (years)
        year_cols = [col for col in bloom_df.columns if col.isdigit()]

        # Melt the dataframe to convert years to rows
        melted = pd.melt(bloom_df,
                         id_vars=['Site Name'],
                         value_vars=year_cols,
                         var_name='Year',
                         value_name='Date')

        # Convert to datetime and filter valid dates
        melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
        return melted.dropna()

    print("Get sakura bloom dates...")
    first_bloom_processed = extract_years_and_dates(first_bloom_df)
    full_bloom_processed = extract_years_and_dates(full_bloom_df)

    # Simplify city names
    # first_bloom_processed['Site Name'] = first_bloom_df['Site Name'].replace('Tokyo Japan', 'Tokyo')
    # full_bloom_processed['Site Name'] = full_bloom_df['Site Name'].replace('Tokyo Japan', 'Tokyo')

    # Merge first and full bloom dates
    bloom_data = pd.merge(
        first_bloom_processed,
        full_bloom_processed,
        on=['Site Name', 'Year'],
        suffixes=('_first', '_full')
    )

    # Convert temps_df date column to datetime
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])

    def get_temp_sequence(row: pd.Series, temp_df: pd.DataFrame,
                          start_date: datetime, end_date: datetime):

        # Convert start_date and end_date to pandas Timestamp
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Create a mask to filter the DataFrame
        mask = (temp_df['Date'] >= start_date) & (temp_df['Date'] <= end_date)

        # Get temperatures for the city using the mask
        temps = temp_df.loc[mask, row['Site Name']].tolist()

        # Interpolate missing values
        # interpolation_methods: Literal['linear', 'spline', 'nearest', 'pchip', 'slinear', 'polynomial'] = 'spline'
        temps = interpolate_values(pd.Series(temps), min_points=30)

        return temps

    def validate_temperature_data(row: pd.Series):
        """
        Temperature lists should have the same length as days_to_first_bloom and days_to_full_bloom (missing values in
        the temperature set are interpolated)
        """
        expected_length_first = row['days_to_first'] + 1 if pd.notna(row['days_to_first']) else 0
        expected_length_full = row['days_to_full'] + 1 if pd.notna(row['days_to_full']) else 0

        # Check if temperature lists exist and have correct lengths
        temps_first_valid = (
                    len(row['temps_to_first']) == expected_length_first) if expected_length_first > 0 else False
        temps_full_valid = (
                    len(row['temps_to_full']) == expected_length_full) if expected_length_full > 0 else False

        # Both lists should be non-empty and have correct lengths
        return temps_first_valid and temps_full_valid

    def process_city_data(group):
        results = []

        for _, row in group.iterrows():
            year = int(row['Year'])
            first_date = row['Date_first']
            full_date = row['Date_full']

            # Calculate start date (previous year)
            start_date = datetime(year - 1, start_month, start_day)

            # Get temperature sequences
            temps_to_first = get_temp_sequence(row, temp_df, start_date, first_date)
            temps_to_full = get_temp_sequence(row, temp_df, start_date, full_date)

            # Calculate days offset
            days_to_first = (first_date - start_date).days
            days_to_full = (full_date - start_date).days

            # Calculate the offset between first and full bloom
            bloom_offset = days_to_full - days_to_first

            # Create countdown sequences
            # countdown_to_first continues to negative values until reaching the full bloom date
            countdown_to_first = list(
                range(days_to_first, -bloom_offset - 1, -1))  # [days_to_first, ..., -bloom_offset]
            countdown_to_full = list(
                range(days_to_full, -1, -1))  # [days_to_full, ..., 0]

            results.append({
                'site_name': row['Site Name'],
                'year': year,
                'first_bloom': first_date,
                'full_bloom': full_date,
                'data_start_date': start_date,
                'days_to_first': days_to_first,
                'days_to_full': days_to_full,
                'bloom_offset': bloom_offset,
                'countdown_to_first': countdown_to_first,
                'countdown_to_full': countdown_to_full,
                'temps_to_first': temps_to_first,
                'temps_to_full': temps_to_full
            })

        return pd.DataFrame(results)

    # Process each city
    print("Get input and output data for reservoir...")
    temp_cities = set(temp_df.columns[1:])
    final_data = pd.DataFrame()
    for city, group in bloom_data.groupby('Site Name'):
        if city in temp_cities:
            print(f"\tProcessing {city}")
            city_data = process_city_data(group)
            final_data = pd.concat([final_data, city_data])
        else:
            print(f"\tSkipping {city}")

    # Remove rows with invalid temperature data
    if drop_inadequate_temps:
        valid_rows = final_data.apply(validate_temperature_data, axis=1)
        final_data = final_data[valid_rows].reset_index(drop=True)

    # Merge with city coordinates
    print("Merge all data...")
    final_data = pd.merge(
        final_data,
        japan_cities,
        left_on='site_name',
        right_on='city_ascii',
        how='left'
    )

    return final_data


def create_sakura_data(start_date: str,  # Begin of Temperature data in "DD:MM"
                       drop_na: bool = True,
                       scale_data: bool = True,
                       file_path: str | None = 'src_training/training_data.parquet') -> (pd.DataFrame, dict):

    # Load datasets
    full_blossom_df = pd.read_csv('./data/sakura_full_bloom_dates.csv')
    first_blossom_df = pd.read_csv('./data/sakura_first_bloom_dates.csv')
    temps_df = pd.read_csv('./data/Japanese_City_Temps.csv')
    city_df = pd.read_csv('./data/worldcities.csv')

    # Process data
    start_day, start_month = parse_date(start_date)

    result_df = process_data(temp_df=temps_df,
                             full_bloom_df=full_blossom_df,
                             first_bloom_df=first_blossom_df,
                             cities_df=city_df,
                             start_day=start_day,
                             start_month=start_month)

    if drop_na:
        result_df = result_df.dropna()

    scalers = {}
    if scale_data:
        for column, config in scalers_config.items():
            result_df, scalers[column] = scale_column(
                df=result_df,
                column_name=config['column_name'],
                scaler_type=config['scaler_type'],
                scaler_kwargs=config['scaler_kwargs']
            )

    print('Saving data...')
    if file_path is not None:
        folder, _ = os.path.split(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        # save scalers
        with open(folder + '/scalers.pickle', 'wb') as f:
            pickle.dump(scalers, f)
        # save dataframe
        result_df.to_parquet(file_path, index=False)

    return result_df, scalers


def load_sakura_data(file_path: str = 'src_training/training_data.parquet') -> (pd.DataFrame, dict):
    # if parquet file exists load it ...
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
        with open(os.path.split(file_path)[0] + '/scalers.pickle', 'rb') as f:
            scalers = pickle.load(f)
    # ... otherwise create it
    else:
        df, scalers = create_sakura_data(start_date="01:08", file_path=file_path)
    return df, scalers


if __name__ == '__main__':

    df, _ = load_sakura_data()
    print(df.head())
    print(df.shape)

    print(np.amin(df['lat'].values))
