import numpy as np
import os


def safe_save(save_name: str, array: np.ndarray) -> None:
    """
    If a folder is specified and does not yet exist, it will be created automatically.
    :param save_name: full path + data name
    :param array: array to save
    :return:
    """
    # create folder if not exists
    folder, data_name = os.path.split(save_name)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if data_name[-3:] == 'npy':
        np.save(save_name, array)
    else:
        np.save(save_name + '.npy', array)


def find_latest_date(df):
    """
    Find the latest date from all year columns in the dataset.
    Excludes non-date columns and converts dates to day of year for comparison.
    """
    import pandas as pd
    from datetime import datetime

    # Read all columns except the known non-date columns
    exclude_cols = ['Site Name', 'Currently Being Observed', '30 Year Average 1981-2010', 'Notes']
    date_cols = [col for col in df.columns if col not in exclude_cols]

    # Convert all valid dates to datetime objects
    latest_date = None
    latest_day_of_year = 0

    for col in date_cols:
        # Skip empty columns or non-date columns
        if df[col].empty or not isinstance(df[col].iloc[0], str):
            continue

        # Convert column to datetime
        dates = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

        # Find the latest date based on day of year
        for date in dates:
            if pd.notna(date):
                day_of_year = date.timetuple().tm_yday
                if day_of_year > latest_day_of_year:
                    latest_day_of_year = day_of_year
                    latest_date = date

    return latest_date, latest_day_of_year

