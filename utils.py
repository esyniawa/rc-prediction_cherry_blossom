import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


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


def calculate_date_difference(df: pd.DataFrame, start_date_col: str, end_date_col: str) -> pd.Series:
    """
    Calculate the difference in days between two date columns in a pandas DataFrame.
    """
    # Convert columns to datetime if they aren't already
    df[start_date_col] = pd.to_datetime(df[start_date_col])
    df[end_date_col] = pd.to_datetime(df[end_date_col])

    # Calculate the difference in days
    days_difference = (df[end_date_col] - df[start_date_col]).dt.days

    return days_difference


def get_list_length_stats(df: pd.DataFrame, list_column: str):
    """
    Calculate the minimum and maximum length of lists in a DataFrame column.
    """
    # Calculate lengths of all lists in the column
    list_lengths = df[list_column].apply(len)

    # Get minimum and maximum lengths
    min_length = list_lengths.min()
    max_length = list_lengths.max()

    return min_length, max_length


def plot_mae_results(predictions_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create bar and line plots of MAE by site and year
    """
    # Set style
    sns.set_theme()

    # Create site plot
    fig_site, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Get unique sites
    sites = sorted(predictions_df['site_name'].unique())
    x = np.arange(len(sites))
    width = 0.35

    # Calculate site statistics
    site_stats = predictions_df.groupby('site_name').agg({
        'mae_first': ['mean', 'std', 'count'],
        'mae_full': ['mean', 'std', 'count']
    })

    # Calculate standard errors
    site_stats['mae_first', 'sem'] = site_stats['mae_first', 'std'] / np.sqrt(site_stats['mae_first', 'count'])
    site_stats['mae_full', 'sem'] = site_stats['mae_full', 'std'] / np.sqrt(site_stats['mae_full', 'count'])

    # Plot first bloom MAE by site
    ax1.bar(x, site_stats['mae_first', 'mean'], width,
            yerr=site_stats['mae_first', 'sem'],
            capsize=5)
    ax1.set_ylabel('MAE (days)')
    ax1.set_title('First Bloom Prediction Error by Site')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sites, rotation=45, ha='right')

    # Plot full bloom MAE by site
    ax2.bar(x, site_stats['mae_full', 'mean'], width,
            yerr=site_stats['mae_full', 'sem'],
            capsize=5)
    ax2.set_ylabel('MAE (days)')
    ax2.set_title('Full Bloom Prediction Error by Site')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sites, rotation=45, ha='right')

    plt.tight_layout()

    # Save site plot if path provided
    if save_path:
        site_path = save_path + '_sites.pdf'
        fig_site.savefig(site_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    # Create year plot
    fig_year, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate year statistics
    year_stats = predictions_df.groupby('year').agg({
        'mae_first': ['mean', 'std', 'count'],
        'mae_full': ['mean', 'std', 'count']
    })

    # Calculate confidence intervals (95%)
    year_stats['mae_first', 'ci'] = 1.96 * year_stats['mae_first', 'std'] / np.sqrt(year_stats['mae_first', 'count'])
    year_stats['mae_full', 'ci'] = 1.96 * year_stats['mae_full', 'std'] / np.sqrt(year_stats['mae_full', 'count'])

    years = sorted(predictions_df['year'].unique())

    # Plot first bloom MAE by year
    ax3.plot(years, year_stats['mae_first', 'mean'], 'b-', label='Mean MAE')
    ax3.fill_between(years,
                     year_stats['mae_first', 'mean'] - year_stats['mae_first', 'ci'],
                     year_stats['mae_first', 'mean'] + year_stats['mae_first', 'ci'],
                     alpha=0.2,
                     label='95% CI')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('MAE (days)')
    ax3.set_title('First Bloom Prediction Error by Year')
    ax3.legend()

    # Plot full bloom MAE by year
    ax4.plot(years, year_stats['mae_full', 'mean'], 'b-', label='Mean MAE')
    ax4.fill_between(years,
                     year_stats['mae_full', 'mean'] - year_stats['mae_full', 'ci'],
                     year_stats['mae_full', 'mean'] + year_stats['mae_full', 'ci'],
                     alpha=0.2,
                     label='95% CI')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('MAE (days)')
    ax4.set_title('Full Bloom Prediction Error by Year')
    ax4.legend()

    plt.tight_layout()

    # Save year plot if path provided
    if save_path:
        year_path = save_path + '_years.pdf'
        fig_year.savefig(year_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close('all')
