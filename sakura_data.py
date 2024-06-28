# Here, the necessary datasets for the sakura blossom date prediciton are processed.
# load_sakura_data function reads the data, pre-processes them and returns features and labels.
# See comments below for more information.
import os.path

# Datasets are downloaded from Kaggle:
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-cherry-blossom-data,
# https://www.kaggle.com/datasets/ryanglasnapp/japanese-temperature-data
# https://www.kaggle.com/datasets/juanmah/world-cities

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def scale_column(df: pd.DataFrame, column_name: str, a: float = 1.):

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

    return df


def create_sakura_data(first_season: int = 1956,
                       last_season: int = 2020,  # Extend this range if more data is available
                       scale_data: float | None = 1.,
                       save_data: bool = True):

    # Load datasets
    blossom_df = pd.read_csv('./data/sakura_full_bloom_dates.csv')
    temps_df = pd.read_csv('./data/Japanese_City_Temps.csv')
    city_df = pd.read_csv('./data/worldcities.csv')

    # Convert temps_df date column to datetime
    temps_df['Date'] = pd.to_datetime(temps_df['Date'])

    # Get the list of cities present in both datasets
    blossom_cities = set(blossom_df['Site Name'])
    temp_cities = set(temps_df.columns) - {'Date'}  # Exclude the 'Date' column
    common_cities = list(blossom_cities.intersection(temp_cities))

    # Function to get temperatures for a city between two dates
    def get_temps(city: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
        if pd.isnull(start_date):
            start_date = end_date - timedelta(days=365)

        if pd.isnull(end_date):
            end_date = start_date + timedelta(days=365)

        mask = (temps_df['Date'] >= start_date) & (temps_df['Date'] < end_date)
        return temps_df.loc[mask, city].tolist()

    # List to store all rows
    result_rows = []

    # Process each common city and year
    for city in common_cities:
        city_data = blossom_df[blossom_df['Site Name'] == city]
        for index, row in city_data.iterrows():
            for year in range(first_season, last_season):

                start_date = pd.to_datetime(row[str(year - 1)])
                end_date = pd.to_datetime(row[str(year)])

                temps = get_temps(city, start_date, end_date)

                # just append if I have enough temperatures
                if len(temps) > 0:
                    avg_temp = np.mean(temps)

                    new_row = {
                        'City': city,
                        'Season': year,
                        'Blossom': end_date.dayofyear,
                        'Mean_Temp': avg_temp,
                        'Temps': np.array(temps)
                    }

                    result_rows.append(new_row)

    # Create the result DataFrame
    result_df = pd.DataFrame(result_rows)

    # Merge with geographical informations
    city_df = city_df[['city_ascii', 'lat', 'lng']]
    # Rename columns
    city_df = city_df.rename(columns={'lat': 'Lat', 'lng': 'Lng'})
    result_df = pd.merge(result_df, city_df, left_on='City', right_on='city_ascii')
    result_df.drop(columns='city_ascii', inplace=True)

    if scale_data is not None:
        for column in ['Temps', 'Mean_Temp', 'Lat', 'Lng']:
            result_df = scale_column(df=result_df, column_name=column, a=scale_data)

    if save_data:
        result_df.to_parquet('data/training_data.parquet', index=False)

    return result_df


def load_sakura_data(file_path: str = 'data/training_data.parquet'):
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    else:
        df = create_sakura_data()
    return df


if __name__ == '__main__':

    df = load_sakura_data()
    print(df.head())
    print(df.shape)

    print(np.amin(df['Lat'].values))
