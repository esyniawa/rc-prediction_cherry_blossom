# Here, the necessary datasets for the sakura blossom date prediciton are processed.
# load_sakura_data function reads the data, pre-processes them and returns features and labels.
# See comments below for more information.
import os.path
import pickle
from calendar import month

# Datasets are downloaded from Kaggle:
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-cherry-blossom-data,
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-temperature-data
# https://www.kaggle.com/datasets/juanmah/world-cities

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


def parse_date(date_str: str, separator: str = ':'):
    day, month = map(int, date_str.split(separator))
    return day, month


def scale_column(df: pd.DataFrame, column_name: str, a: float = 1.):
    """
    Function to scale a column of a dataframe weather it is a scalar or an array
    :param df: Dataframe
    :param column_name: Column name to scale
    :param a: Min and max value after the scaling
    :return: Scaled dataframe
    """
    # Function to flatten arrays or convert scalars to arrays
    def process_element(x):
        return np.atleast_1d(x).flatten()

    # Process and flatten all elements in the column
    flattened = np.concatenate(df[column_name].apply(process_element))

    # Create and fit the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-a, a))
    scaler.fit(flattened.reshape(-1, 1))

    # Define a function to scale individual elements
    def scale_element(x):
        if np.isscalar(x):
            return scaler.transform([[x]])[0][0]
        else:
            return scaler.transform(x.reshape(-1, 1)).flatten()

    # Apply the scaling function to each element in the column
    df[column_name] = df[column_name].apply(scale_element)

    return df, scaler


def process_data(temp_df: pd.DataFrame,
                 first_bloom_df: pd.DataFrame,
                 full_bloom_df: pd.DataFrame,
                 cities_df: pd.DataFrame,
                 start_month: int = 7,
                 start_day: int = 1) -> pd.DataFrame:
    """
    Function to merge the different datasets

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
    first_bloom_processed['Site Name'] = first_bloom_df['Site Name'].replace('Tokyo Japan', 'Tokyo')
    full_bloom_processed['Site Name'] = full_bloom_df['Site Name'].replace('Tokyo Japan', 'Tokyo')

    # Merge first and full bloom dates
    bloom_data = pd.merge(
        first_bloom_processed,
        full_bloom_processed,
        on=['Site Name', 'Year'],
        suffixes=('_first', '_full')
    )

    # Convert temps_df date column to datetime
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])

    def get_temp_sequence(row, temp_df, start_date, end_date):
        # Convert start_date and end_date to pandas Timestamp
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Create a mask to filter the DataFrame
        mask = (temp_df['Date'] >= start_date) & (temp_df['Date'] <= end_date)

        # Get temperatures for the city using the mask
        temps = temp_df.loc[mask, row['Site Name']].tolist()

        # Interpolate missing values
        temps = pd.Series(temps).interpolate(method='linear').tolist()
        print(len(temps))
        return temps

    def process_city_data(group):
        results = []

        for _, row in group.iterrows():
            year = int(row['Year'])
            first_date = row['Date_first']
            full_date = row['Date_full']

            # Calculate start date (previous year)
            start_date = datetime(year - 1, start_month, start_day)

            # debug ...
            print(f"Start Date: {start_date}")
            print(f"First Bloom Date: {first_date}")
            print(f"Full Bloom Date: {full_date}")
            print(f"temp_df shape: {temp_df.shape}")
            print(f"temp_df columns: {temp_df.columns}")
            print(f"temp_df Date range: {temp_df['Date'].min()} - {temp_df['Date'].max()}")


            # Get temperature sequences
            #TODO: This function returns only empty lists
            temps_to_first = get_temp_sequence(row, temp_df, start_date, first_date)
            temps_to_full = get_temp_sequence(row, temp_df, start_date, full_date)

            # Calculate days offset
            days_to_first = (first_date - start_date).days
            days_to_full = (full_date - start_date).days

            results.append({
                'site_name': row['Site Name'],
                'year': year,
                'first_bloom': first_date,
                'full_bloom': full_date,
                'days_to_first': days_to_first,
                'days_to_full': days_to_full,
                'temps_to_first': temps_to_first,
                'temps_to_full': temps_to_full
            })

        return pd.DataFrame(results)

    # Process each city
    print("Get input and output data...")
    temp_cities = set(temp_df.columns[1:])
    final_data = pd.DataFrame()
    for city, group in bloom_data.groupby('Site Name'):
        if city in temp_cities:
            print(f"\tProcessing {city}, Year: {group['Year'].values}")
            city_data = process_city_data(group)
            final_data = pd.concat([final_data, city_data])
        else:
            print(f"\tSkipping {city}")

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
                       scale_data: float | None = 1.,
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
    if scale_data is not None:
        for column in ['days_to_first', 'days_to_full', 'temps_to_first', 'temps_to_full', 'lat', 'lng']:
            result_df, scalers[column] = scale_column(df=result_df, column_name=column, a=scale_data)

    print('Saving data...')
    if file_path is not None:
        folder, _ = os.path.split(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(folder + 'scalers.pickle', 'w') as f:
            pickle.dump(scalers, f)

        result_df.to_parquet(file_path, index=False)

    return result_df, scalers


def load_sakura_data(file_path: str = 'src_training/training_data.parquet') -> (pd.DataFrame, dict):
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
        with open(os.path.split(file_path)[0] + 'scalers.pickle', 'r') as f:
            scalers = pickle.load(f)
    else:
        df, scalers = create_sakura_data(start_date="01:07", file_path=file_path)
    return df, scalers


if __name__ == '__main__':

    df, _ = load_sakura_data()
    print(df.head())
    print(df.shape)

    print(np.amin(df['Lat'].values))
