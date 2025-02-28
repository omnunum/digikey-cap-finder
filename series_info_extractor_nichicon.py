import os
import pandas as pd
import requests
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
        for col in series_df.columns:
            series_df[col] = series_df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
            series_df[col] = series_df[col].astype(str).str.replace(r' \(.+\)', '', regex=True)
            series_df[col] = series_df[col].astype(str).str.replace(',', '')

        # Save to CSV
        output_file = os.path.join(output_dir, f"{series}_Standard Ratings.csv")
        cols_to_drop = [
            'Purchase', 'All', 'Series', 'Low temperature ESR standard',
            'AEC-Q200 compliant', 'Audio equipment', 'Capacitance Tolerance (%)'
        ]
        for col in cols_to_drop:
            if col in series_df.columns:
                series_df = series_df.drop(col, axis=1)
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