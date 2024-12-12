import openpyxl
import pandas as pd
import csv

def convert_to_csv(file_path, output_csv):
    # Read all sheets into a dictionary of DataFrames
    excel_data = pd.read_excel(file_path, sheet_name=None, header=None)

    combined_data = []

    # Iterate through sheets and extract city names and data
    for sheet_name, sheet_data in excel_data.items():
        # Locate the row with 'Year' as the starting point for data
        year_row_idx = sheet_data.iloc[:, 0][sheet_data.iloc[:, 0].astype(str).str.contains("Year", na=False)].index[0]
        data_start = sheet_data.iloc[year_row_idx:]

        # Set the first row as column headers
        data_start.columns = data_start.iloc[0]
        data_start = data_start[1:]

        # Add a city column based on the sheet name or metadata
        data_start["City"] = sheet_name

        # Filter rows from 1969 onward
        data_start = data_start[data_start["Year"].astype(str).str.isdigit()]
        data_start["Year"] = data_start["Year"].astype(int)
        data_start = data_start[data_start["Year"] >= 1969]

        combined_data.append(data_start)

    # Combine all sheets
    combined_data = pd.concat(combined_data, ignore_index=True)

    # Save to CSV
    combined_data.to_csv(output_csv, index=False, encoding='utf-8')

if __name__ == "__main__":
    file_path = "sorted.xlsx"
    output_csv = "humidity_data.csv"
    convert_to_csv(file_path, output_csv)
