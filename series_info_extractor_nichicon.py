import os
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from helpers import make_cached_request

def fetch_nichicon_data():
    """Fetch Nichicon capacitor data from their website."""
    url = "https://www.nichicon.co.jp/english/products/aec/"
    
    params = {
        'per_page': '10000'
    }
    
    payload = {
        'series': '',
        'part_number': '',
        'type[]': 'TP04',
        'hidden-s-endurance': 'false',
        's-endurance_lower': '0',
        's-endurance_upper': '20000',
        'hidden-s-rated_voltage': 'false',
        's-rated_voltage_lower': '2.5',
        's-rated_voltage_upper': '630',
        'hidden-s-rated_capacitance_micro': 'false',
        's-rated_capacitance_micro_lower': '0.1',
        's-rated_capacitance_micro_upper': '680000',
        'hidden-s-product_diameter': 'false',
        's-product_diameter_lower': '4',
        's-product_diameter_upper': '100',
        'hidden-s-product_height': 'false',
        's-product_height_lower': '3.9',
        's-product_height_upper': '250',
        'hidden-s-rated_ripple_120hz': 'false',
        's-rated_ripple_120hz_lower': '1',
        's-rated_ripple_120hz_upper': '48200',
        'hidden-s-rated_ripple_100khz': 'false',
        's-rated_ripple_100khz_lower': '6',
        's-rated_ripple_100khz_upper': '8100',
        'hidden-s-high_temperature_impedance': 'false',
        's-high_temperature_impedance_lower': '10',
        's-high_temperature_impedance_upper': '11000',
        'hidden-s-high_temperature_esr': 'false',
        's-high_temperature_esr_lower': '5',
        's-high_temperature_esr_upper': '2800',
        'search_product': '',
        'search-product': 'true'
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0'
    }
    
    response_text = make_cached_request(
        url=url,
        payload=payload,
        headers=headers,
        cache_dir='cache',
        response_type='html',
        params=params
    )
    
    # Parse HTML
    soup = BeautifulSoup(response_text, 'html.parser')
    table = soup.find('table', {'id': 'searchProductsResultTable'})
    
    if table is None:
        print("Table not found in response")
        return None
        
    # Extract headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())
        
    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header row
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        if row:  # Only add non-empty rows
            rows.append(row)
            
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
        
    return df

def process_nichicon_series(df: pd.DataFrame, output_dir: str = 'series_tables/nichicon'):
    """
    Process Nichicon data and save each series to its own directory.
    
    Args:
        df: DataFrame containing Nichicon capacitor data
        output_dir: Base directory for output files
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique series
    series_list = df['Series'].unique()
    
    # Process each series
    for series in series_list:
        # Filter data for this series
        series_df = df[df['Series'] == series].copy()
        # Clean values of common issues
        # Define columns that need mΩ to Ω conversion
        milliohm_columns = [
            'Impedance (mΩ)(-10℃/100kHz)', 
            'Impedance (mΩ)(20℃/100kHz)',
            'ESR(mΩ)（20℃/100kHz）',
            'ESR(mΩ) (-40℃/100kHz）',
            'ESR(mΩ) after endurance test'
        ]
        
        ripple_columns = [
            'Rated Ripple1(mArms)',
            'Rated Ripple2(mArms)'
        ]
        
        cols_to_drop = [
            'Purchase', 'All', 'Series', 'Low temperature ESR standard',
            'AEC-Q200 compliant', 'Audio equipment', 'Capacitance Tolerance (%)'
        ]
        
        # First, handle ripple columns to extract rating information
        for col in ripple_columns:
            if col in series_df.columns and not series_df.empty:
                # Get first non-empty value to extract rating
                sample_values = series_df[col].dropna()
                rating = ""
                new_col_name = col
                
                if not sample_values.empty:
                    sample_value = sample_values.iloc[0]
                    # Extract rating in parentheses using regex
                    rating_match = re.search(r'\((.*?)\)', sample_value)
                    rating = rating_match.group(1) if rating_match else ""
                    
                    if rating:
                        # Rename the column to include the rating but preserve the column number
                        new_col_name = f"{col.split('(')[0]}({rating})"
                        series_df.rename(columns={col: new_col_name}, inplace=True)
                        
                        # Update the column reference for further processing
                        ripple_columns[ripple_columns.index(col)] = new_col_name
                
                # Clean the values - remove the rating part
                series_df[new_col_name] = series_df[new_col_name].astype(str).apply(
                    lambda x: re.sub(r'\s*\(.*?\)', '', x) if isinstance(x, str) and '(' in x else x
                )

        for col in series_df.columns:
            # Apply basic cleaning to all columns
            series_df[col] = series_df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
            series_df[col] = series_df[col].astype(str).str.replace(',', '')
            
            # Apply specialized conversion for milliohm columns
            if col in milliohm_columns:
                series_df[col] = pd.to_numeric(series_df[col], errors='coerce') / 1000
                new_col_name = col.replace('(mΩ)', '(Ω)')
                series_df.rename(columns={col: new_col_name}, inplace=True)
            
            # Convert ripple values to numeric after cleaning
            if col in ripple_columns:
                series_df[col] = pd.to_numeric(series_df[col], errors='coerce')
            
            if col in cols_to_drop:
                series_df = series_df.drop(col, axis=1)
        # Save to CSV
        output_file = os.path.join(output_dir, f"{series}_Standard Ratings.csv")
        series_df.to_csv(output_file, index=False)

def main():
    """Main function to fetch and process Nichicon data."""
    # Fetch data
    print("Fetching Nichicon data...")
    df = fetch_nichicon_data()
    
    if df is not None:
        # Process data
        print("Processing Nichicon series...")
        process_nichicon_series(df)
        print("Done! Check the series_tables/nichicon directory for the processed files.")
    else:
        print("Error: Failed to fetch Nichicon data.")

if __name__ == '__main__':
    main()