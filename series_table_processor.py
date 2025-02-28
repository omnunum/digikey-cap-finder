import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileType(Enum):
    RATINGS = "ratings"
    DISSIPATION = "dissipation"
    UNKNOWN = "unknown"

@dataclass
class FileInfo:
    manufacturer: str
    relative_path: str
    raw_headers: List[str]      # Original headers before standardization
    mapped_headers: List[str]  # Headers after standardization
    file_type: FileType
    series_name: str
    standardized_filename: str  # The standardized filename for output
    df: pd.DataFrame # The processed DataFrame

def map_header_to_standard_name(header: str) -> str:
    """
    Standardize a header using regex patterns.
    Returns the original header if no pattern matches.
    """
    def extract_temp(header: str) -> str:
        """Extract temperature from header if present"""
        temp_match = re.search(r'[\-]?\d+(?:℃|°C|C)', header)
        if temp_match:
            return (temp_match.group(0)
                    .replace('°C', '℃')
                    .replace('C', '℃')
                    .replace('℃', '°C')
                    .replace(' ', ''))
        return ''

    def extract_frequency(header: str) -> str:
        """Extract frequency from header if present"""
        freq_match = re.search(r'\d{2,}\s?(Hz|kHz)', header)
        if freq_match:
            return freq_match.group(0).replace(' ', '')
        return ''

    def extract_temp_freq(header: str) -> str:
        """Extract temperature and frequency from header if present"""
        temp = extract_temp(header)
        freq = extract_frequency(header)
        if temp and freq:
            return f"{temp}@{freq}"
        elif temp:
            return temp
        elif freq:
            return f"@{freq}"
        return ''
    
    patterns = {
        # Basic electrical parameters
        r".*?cap(?:acitance)?.*": "Capacitance", 
        r".*?esr.*": lambda x: f"ESR {extract_temp_freq(x)}".strip(),
        r".*?esl.*": lambda x: f"ESL {extract_temp_freq(x)}".strip(),
        r".*?(?:impedance|^z$).*": lambda x: f"Impedance {extract_temp_freq(x)}".strip(),
        r".*?max.*?ripple.*$": lambda x: f"Max Ripple Current {extract_temp_freq(x)}".strip(),
        r".*?ripple.*|^rc$": lambda x: f"Ripple Current {extract_temp_freq(x)}".strip(),
        r".*?surge.*": "Surge Voltage",
        r".*?(?:voltage|vdc|wv).*": "Voltage",
        r".*?leak.*current.*|^lc$": "Leakage Current",
        r".*?(?:tan.*[δd]|^df$|dissipation.*factor).*": "Dissipation Factor",
        
        # Other parameters
        r".*?endurance.*": lambda x: f"Endurance {extract_temp_freq(x)}".strip(),
        r".*?part.*(?:no|number).*|^number$": "Part Number",
        r".*?(?:min.*packag(e|ing)).*": lambda x: (
            "Min Packaging Qty Straight" if "straight" in x.lower() else
            "Min Packaging Qty Taping"
        ),

        # Physical parameters
        r".*?(?:size).*(?:code).*": "Size Code",
        # Combined dimension pattern - check this first
        r".*?(?:case|size).*(?:[φøo]?d.*[×x].*l|[φøo]?d.*l|l.*[φøo]?d).*": "Case Size",
        # Individual dimension patterns - check these after
        r".*?(?:size|diameter).*(?:[φøo]?d).*": "Case Size Diameter",
        r".*?(?:size|length|height).*(?:[φøo]?l).*": "Case Size Length",
        r".*?(size).*": "Case Size",
        r".*?lead.*dia.*|.*[φøo]d.*lead.*": "Lead Diameter",
        r".*?lead.*space.*": lambda x: (
            "Lead Space Taping" if "taping" in x.lower() else
            "Lead Space"
        ),
        r"^(?:[φøo]?l).*": "Case Size Length",
        r"^(?:[φøo]?d).*": "Case Size Diameter",
    }
    
    # Apply patterns to the header
    for pattern, replacement in patterns.items():
        if not re.match(pattern, header, flags=re.IGNORECASE):
            continue
        if callable(replacement):
            return str(replacement(header))
        else:
            return str(replacement)
    return header

def classify_and_standardize_filename(filepath: str) -> Tuple[str, FileType]:
    """
    Standardize the filename by:
    1. Converting series to lowercase
    2. Using standard formats: {series}_ratings.csv or {series}_dissipation.csv
    
    Returns:
        Tuple[str, FileType]: (standardized filename, file_type)
    """
    path = Path(filepath)
    
    # Get series name from the filename (everything before the first underscore or space)
    series = path.stem.split('_')[0].split()[0].lower()
    
    if any(term in path.name.lower() for term in [
        'dissipation',
        'factor',
        'tanδ',
        'df',
        'tangent of loss',
        'tan δ',
        'tan d'
    ]):
        return f"{series}_dissipation.csv", FileType.DISSIPATION
    
    # Check if this is a standard ratings file
    if any(term in path.name.lower() for term in [
        'standard size',
        'standard rating',
        'ratings list',
        'characteristics list',
        'capacitor characteristics'
    ]):
        return f"{series}_ratings.csv", FileType.RATINGS
    
    # For other files (like ripple current specs), keep original name but standardize series
    rest_of_name = '_'.join(path.stem.split('_')[1:])
    if rest_of_name:
        return f"{series}_{rest_of_name}.csv", FileType.UNKNOWN
    return path.name.lower(), FileType.UNKNOWN

def convert_dissipation_table_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a dissipation factor table with voltage columns into a two-column format.
    Input format:
        Voltage | 6.3V | 10V | 16V | ...
        100uF  | 0.12 | 0.14| 0.15| ...
    Output format:
        Voltage | Dissipation Factor
        6.3V   | 0.12
        10V    | 0.14
        16V    | 0.15
    """
    # Skip if already in correct format
    has_tan_column = re.search(r'dissipation|tanδ|df|tan', str(list(df.columns)[1]).lower())
    if has_tan_column:
        return df
        
    # Pivot the dataframe to create voltage/dissipation pairs
    melted = pd.melt(
        df,
        id_vars=[],
        value_vars=list(df.columns)[1:],
        var_name='Voltage',
        value_name='Dissipation Factor'
    )
    
    # Drop any rows where dissipation factor is NaN
    melted = melted.dropna(subset=['Dissipation Factor'])
    
    return melted

def find_csv_files(input_dir: str) -> List[str]:
    """
    Find all CSV files in the input directory.
    
    Args:
        input_dir: Directory containing input CSV files
        
    Returns:
        List of paths to CSV files
    """
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def remove_empty_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are all NaN or empty values.
    
    Args:
        input_df: DataFrame to clean
        
    Returns:
        A new DataFrame with empty columns removed
    """
    df = input_df.copy()
    
    all_na_cols = [col for col in df.columns if df[col].isna().all()]
    all_empty_cols = [col for col in df.columns if (df[col].astype(str).str.strip() == '').all()]    
    cols_to_drop = list(set(all_na_cols + all_empty_cols))
    
    if cols_to_drop:
        logger.debug(f"Dropping {len(cols_to_drop)} empty columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df

def clean_and_convert_column_values(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize values in the dataframe based on column type.
    - Replace placeholder values (dashes, N/A, etc.) with None using regex
    - Remove commas and quotes from all columns
    - For numerical columns, extract numbers and convert to appropriate type
    - Apply specific standardization for known columns
    - Break Case Size columns into diameter (D) and length (L) components
    
    Args:
        input_df: DataFrame to clean and convert
        
    Returns:
        A new DataFrame with standardized values
    """
    df = input_df.copy()
    
    numerical_columns = [
        'Capacitance', 
        'ESR', 
        'Impedance', 
        'Ripple Current',
        'Surge Voltage',
        'Leakage Current',
        'Dissipation Factor',
        'Frequency',
        'Case Size Diameter',
        'Case Size Length'
    ]
    
    placeholder_pattern = r'^(\s*|—|–|-|\.{1,3}|N/?A|n/?a|NA|na)$'
    
    i = 0
    while i < len(df.columns):
        col = df.columns[i]
        
        # Skip processing if all values are NaN
        if df[col].isna().all():
            i += 1
            continue
        df[col] = df[col].astype(str).replace(placeholder_pattern, np.nan, regex=True)
        df[col] = df[col].str.replace('[,"]*', '', regex=True)
        
        if col == 'Case Size':
            # Extract both diameter and length in one regex operation
            # Common formats: "5x11", "φ5×11", "5.0×11.0", "φ5.0×11.0"
            match = df[col].str.extract(r'(\d+\.?\d*)\s*[×x]\s*(\d+\.?\d*)', expand=True)
            
            # Verify that the matches are not all NaN for both columns
            if not match.isna().all().all():
                # Add new columns to the DataFrame
                df['Case Size Diameter'] = pd.to_numeric(match[0], errors='coerce')
                df['Case Size Length'] = pd.to_numeric(match[1], errors='coerce')
            else:
                logger.warning(f"No DxL matches found for Case Size in column {col}")
            df.drop(columns=[col], inplace=True)
            continue
        elif col == 'Voltage':
            # Voltage is a special case because it can have ranges like "100V-245V"
            #  that will get expanded into multiple rows later
            df[col] = df[col].str.replace(r'([vV])', '', regex=True)
        elif col in numerical_columns:
            # Extract numerical values and convert to numeric
            df[col] = df[col].str.extract(r'(\d+(?:\.\d+)?)', expand=False)
            # First convert to numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        i += 1
    
    return df

def make_column_names_unique(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names unique by appending numbers to duplicate names.
    For example, if there are two columns named "Voltage", the second one 
    will be renamed to "Voltage_1".
    
    Args:
        input_df: DataFrame with possibly duplicate column names
        
    Returns:
        A new DataFrame with unique column names
    """
    df = input_df.copy()
    
    # If all columns are unique, return the original dataframe
    if len(df.columns) == len(set(df.columns)):
        return df
    
    col_counts: Counter = Counter()
    new_columns = []
    
    for col in df.columns:
        if col_counts[col] == 0:
            new_columns.append(col)
        else:
            new_columns.append(f"{col}_{col_counts[col]}")
        col_counts[col] += 1
    
    df.columns = pd.Index(new_columns)
    logger.debug(f"After handling duplicates: {list(df.columns)}")
    
    return df

def parse_and_standardize_file(input_path: str) -> Optional[FileInfo]:
    """
    Process a single file: read, pivot if needed, standardize headers.
    
    Args:
        input_path: Path to the input CSV file
    
    Returns:
        FileInfo object containing the processed DataFrame and metadata, or None if file type is unknown
    """
    manufacturer = os.path.basename(os.path.dirname(input_path))
    
    # Classify the file by examining its name
    file_name = os.path.basename(input_path)
    std_filename, file_type = classify_and_standardize_filename(file_name)
    series_name = Path(std_filename).stem.split('_')[0]
    
    # Skip unknown file types
    if file_type == FileType.UNKNOWN:
        logger.warning(f"File {input_path} is neither a ratings nor a dissipation file. Skipping.")
        return None
    
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='error')
    logger.debug(f"Processing file: {input_path}")
    logger.debug(f"Original columns: {list(df.columns)}")
    
    df = remove_empty_columns(df)
    raw_headers = list(df.columns)

    # If this is a dissipation factor table, pivot it
    if file_type == FileType.DISSIPATION:
        df = convert_dissipation_table_to_standard_format(df)
    
    df = df.rename(columns=lambda x: map_header_to_standard_name(x))
    logger.debug(f"After standardizing headers: {list(df.columns)}")
    
    # Make column names unique if there are duplicates
    df = make_column_names_unique(df)
    
    mapped_headers = list(df.columns)
    
    df = clean_and_convert_column_values(df)
    logger.debug(f"After standardizing values: {list(df.columns)}")

    # Create file info with the DataFrame embedded
    file_info = FileInfo(
        manufacturer=manufacturer,
        relative_path=input_path,
        raw_headers=raw_headers,
        mapped_headers=mapped_headers,
        file_type=file_type,
        series_name=series_name,
        standardized_filename=std_filename,
        df=df
    )
    
    return file_info

def calculate_low_frequency_esr(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ESR from Dissipation Factor and Capacitance.
    
    Args:
        input_df: DataFrame containing 'Dissipation Factor' and 'Capacitance' columns
        
    Returns:
        A new DataFrame with added 'ESR 20°C@120Hz' column
    """
    # Create a copy of the DataFrame to avoid modifying the input
    df = input_df.copy()
    
    if 'Dissipation Factor' in df.columns and 'Capacitance' in df.columns:
        # ESR = DF / (2π × f × C)
        # where:
        # - DF is Dissipation Factor (unitless)
        # - f is frequency in Hz (120Hz standard for electrolytic capacitors)
        # - C is capacitance in Farads
        # 
        # Since capacitance values in datasheets are typically in microfarads (μF),
        # we need to convert to Farads by multiplying by 1e-6 (1 μF = 10^-6 F)
        def compute_esr_value(row):
            if pd.notnull(row['Dissipation Factor']) and pd.notnull(row['Capacitance']) and row['Capacitance'] > 0:
                return round(row['Dissipation Factor'] / (2 * 3.14159 * 120 * row['Capacitance'] * 1e-6), 3)
            else:
                return None
        
        df['ESR 20°C@120Hz'] = df.apply(compute_esr_value, axis=1)
    
    return df

def process_ratings_with_dissipation(ratings_df: pd.DataFrame, dissipation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a series that has both ratings and dissipation data.
    Merges the dissipation data into the ratings table and computes ESR.
    
    Args:
        ratings_df: DataFrame containing ratings data
        dissipation_df: DataFrame containing dissipation data
        
    Returns:
        DataFrame with merged data and computed ESR
    """
    # Create copies to avoid modifying the input DataFrames
    ratings_df = ratings_df.copy()
    dissipation_df = dissipation_df.copy()
    
    expanded_dissipation_df = resolve_voltage_ranges_to_specific_values(dissipation_df, ratings_df)
    
    # Ensure Voltage is properly formatted for joining
    ratings_df['Voltage'] = ratings_df['Voltage'].astype(float)
    expanded_dissipation_df['Voltage'] = expanded_dissipation_df['Voltage'].astype(float)

    # Left join to keep all ratings rows
    merged_df = pd.merge(
        ratings_df,
        expanded_dissipation_df[['Voltage', 'Dissipation Factor']],
        on='Voltage',
        how='left'
    )
     
    merged_df = calculate_low_frequency_esr(merged_df)
    if merged_df['Dissipation Factor'].isna().any():
        pass
    return merged_df

def process_series_tables_by_type(file_infos: List[FileInfo]) -> List[FileInfo]:
    """
    Second pass: Join dissipation data into ratings tables, standardize values,
    compute ESR, and reorder columns.
    
    Args:
        file_infos: List of FileInfo objects
        
    Returns:
        List of FileInfo objects to be saved (dissipation files that were merged will have df=None)
    """
    series_groups = defaultdict(list)
    for file_info in file_infos:
        series_groups[file_info.series_name].append(file_info)
    
    # Process each series
    for series_name, group in series_groups.items():
        ratings_info = None
        dissipation_info = None
        
        for file_info in group:
            if file_info.file_type == FileType.RATINGS:
                ratings_info = file_info
            elif file_info.file_type == FileType.DISSIPATION:
                dissipation_info = file_info
        
        # Process files based on what we have
        if ratings_info:
            if dissipation_info and "Dissipation Factor" not in ratings_info.df.columns:
                # We have both ratings and dissipation, and need to merge in dissipation data
                ratings_info.df = process_ratings_with_dissipation(ratings_info.df, dissipation_info.df)
            else:
                # Only ratings data
                ratings_info.df = calculate_low_frequency_esr(ratings_info.df.copy())
    
    return file_infos

def resolve_voltage_ranges_to_specific_values(dissipation_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand voltage ranges in dissipation tables by matching with actual voltages from ratings file.
    
    Args:
        dissipation_df: DataFrame containing dissipation data with possible voltage ranges
        ratings_df: DataFrame containing ratings data with actual voltage values
        
    Returns:
        DataFrame with expanded voltage ranges
    """
    if 'Voltage' not in ratings_df.columns:
        print("Warning: Ratings file does not contain Voltage column")
        return dissipation_df
    
    available_voltages = sorted(ratings_df['Voltage'].unique())
    if not available_voltages:
        print("Warning: No voltage values found in ratings file")
        return dissipation_df
    
    expanded_rows = []
    
    for _, row in dissipation_df.iterrows():
        separators = r'[~～\-/]|to'
        if not re.search(separators, str(row['Voltage'])):
            expanded_rows.append(row.to_dict())
            continue
        
        start_str, end_str = re.split(separators, str(row['Voltage']))

        start_num = float(start_str.strip())
        end_num = float(end_str.strip())
        
        matching_voltages = [v for v in available_voltages if start_num <= float(v) <= end_num]

        if matching_voltages:
            for v in matching_voltages:
                expanded_rows.append(row.to_dict() | {'Voltage': v})
        else:
            expanded_rows.append(row.to_dict())

    return pd.DataFrame(expanded_rows)

def generate_header_mapping_report(file_infos: List[FileInfo]):
    """Print analysis of the processed files."""
    print("\nSeries Tables Analysis")
    print("=" * 80 + "\n")
    
    # Group headers by their mapped values
    header_groups = defaultdict(set)
    
    for file_info in file_infos:
        for mapped, original in zip(file_info.mapped_headers, file_info.raw_headers):
            header_groups[mapped].add(original)
        
    print("Summary Statistics:")
    print(f"Total files: {len(file_infos)}\n")
    
    print("Header Mappings:")
    print("=" * 40)
    
    # Print mapped headers first
    for mapped_name, original_headers in sorted(header_groups.items()):
        if len(original_headers) == 1 and mapped_name == list(original_headers)[0]:
            continue
        print(f"\n{mapped_name}:")
        for header in sorted(original_headers):
            print(f"  - {header}")
    # Print unmapped headers
    print("\nUnmapped Headers:")
    for mapped_name, original_headers in sorted(header_groups.items()):
        if len(original_headers) == 1 and mapped_name == list(original_headers)[0]:
            print(f"  - {mapped_name}")

def generate_esr_coverage_report(output_base_dir: str) -> None:
    """
    Analyze processed files to report on ESR data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
    """
    print("\nESR Data Analysis")
    print("=" * 80)
    
    # Get all processed CSV files
    if not Path(output_base_dir).exists():
        print(f"Error: Output directory {output_base_dir} does not exist")
        return
    
    csv_files = list(Path(output_base_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {output_base_dir}")
        return
    
    # Initialize counters and data structures
    total_files = 0
    files_with_computed_esr = 0
    files_with_other_esr = 0
    files_with_both = 0
    files_with_null_esr = 0
    
    esr_columns_by_file = {}
    null_esr_counts = {}
    
    # Analyze each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            total_files += 1
            
            # Check for computed ESR
            has_computed_esr = 'ESR 20°C@120Hz' in df.columns
            
            # Check for other ESR columns
            other_esr_cols = [col for col in df.columns if 'ESR' in col and col != 'ESR 20°C@120Hz']
            has_other_esr = len(other_esr_cols) > 0
            
            # Update counters
            if has_computed_esr:
                files_with_computed_esr += 1
                
                # Count null values in computed ESR
                null_count = df['ESR 20°C@120Hz'].isna().sum()
                total_count = len(df)
                if null_count > 0:
                    files_with_null_esr += 1
                    null_esr_counts[file_path.stem] = (null_count, total_count)
            
            if has_other_esr:
                files_with_other_esr += 1
                esr_columns_by_file[file_path.stem] = other_esr_cols
            
            if has_computed_esr and has_other_esr:
                files_with_both += 1
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Print summary statistics
    print(f"\nTotal files analyzed: {total_files}")
    print(f"Files with computed ESR: {files_with_computed_esr} ({files_with_computed_esr/total_files*100:.1f}%)")
    print(f"Files with other ESR columns: {files_with_other_esr} ({files_with_other_esr/total_files*100:.1f}%)")
    print(f"Files with both computed and other ESR: {files_with_both} ({files_with_both/total_files*100:.1f}%)")
    print(f"Files with null computed ESR values: {files_with_null_esr} ({files_with_null_esr/files_with_computed_esr*100:.1f}% of files with computed ESR)")
    
    # Print details about files with null ESR values
    if null_esr_counts:
        print("\nFiles with null computed ESR values:")
        print("-" * 60)
        for file_name, (null_count, total_count) in sorted(null_esr_counts.items(), key=lambda x: x[1][0]/x[1][1], reverse=True):
            print(f"{file_name}: {null_count}/{total_count} values null ({null_count/total_count*100:.1f}%)")
    
    # Print details about other ESR columns
    if esr_columns_by_file:
        print("\nFiles with other ESR columns:")
        print("-" * 60)
        for file_name, columns in sorted(esr_columns_by_file.items()):
            print(f"{file_name}: {', '.join(columns)}")

def save_processed_files(file_infos: List[FileInfo], output_dir: str) -> None:
    """
    Save processed files to the output directory.
    Also creates a consolidated CSV with only priority columns from all series.
    
    Args:
        file_infos: List of FileInfo objects to save
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_saved = 0
    files_skipped = 0
    
    priority_cols = ['Voltage', 'Capacitance', 'ESR 20°C@120Hz', 'ESR 20°C@100kHz', 'Impedance 20°C@100kHz', 'Case Size Diameter', 'Case Size Length']
    all_series_data = []
    
    for file_info in file_infos:
        if file_info.file_type == FileType.DISSIPATION:
            files_skipped += 1
            logger.debug(f"Skipping dissipation file: {file_info.relative_path}")
            continue
        
        if file_info.file_type == FileType.UNKNOWN:
            files_skipped += 1
            logger.info(f"Skipping unknown file type: {file_info.relative_path}")
            continue
            
        file_output_path = output_path / file_info.standardized_filename
            
        if file_info.file_type == FileType.RATINGS:
            df = file_info.df.copy()
            
            available_priority_cols = [col for col in priority_cols if col in df.columns]
            other_cols = sorted([col for col in df.columns if col not in priority_cols])
            df = df[available_priority_cols + other_cols]
            
            # Process numeric columns to remove trailing zeros
            for col in available_priority_cols:
                if col in df.columns:
                    # Convert to string, replace trailing zeros, and convert NaN to empty string
                    df[col] = df[col].astype(str).replace(r'\.0$', '', regex=True).replace('nan', '')
            df = df.fillna('')
            
            # Save the individual file
            df.to_csv(file_output_path, index=False)
            files_saved += 1
            logger.debug(f"Saved {file_info.standardized_filename}")
            
            # Add metadata for consolidated file
            df['Series'] = file_info.series_name
            df['Manufacturer'] = file_info.manufacturer
            all_series_data.append(df)
    
    if all_series_data:
        consolidated_df = pd.concat(all_series_data, ignore_index=True)
        
        # Create the merged ESR/Z column by coalescing ESR and Impedance
        consolidated_df['ESR/Z 20°C@100kHz'] = consolidated_df['ESR 20°C@100kHz'].combine_first(consolidated_df['Impedance 20°C@100kHz'])
        consolidated_df = consolidated_df.drop(columns=['ESR 20°C@100kHz', 'Impedance 20°C@100kHz'], errors='ignore')
        
        # Define columns for consolidated output
        output_cols = ['Series', 'Manufacturer', 'Voltage', 'Capacitance', 'ESR 20°C@120Hz', 'ESR/Z 20°C@100kHz', 'Case Size Diameter', 'Case Size Length']
        
        # Keep only columns that exist in the dataframe
        available_cols = [col for col in output_cols if col in consolidated_df.columns]
        consolidated_df = consolidated_df[available_cols]
        
        # Ensure all NaN values are converted to empty strings
        consolidated_df = consolidated_df.fillna('')
        
        consolidated_output_path = output_path / "all_series_priority_data.csv"
        consolidated_df.to_csv(consolidated_output_path, index=False)
        logger.info(f"Saved consolidated priority data to {consolidated_output_path}")
    
    logger.info(f"Saved {files_saved} files, skipped {files_skipped} files")

def main():
    """
    Main function that orchestrates the entire data processing pipeline.
    """
    # Configuration
    input_dir = "series_tables"
    output_dir = "series_tables_processed"
    
    # Step 1: Find all CSV files
    print("Finding CSV files...")
    input_files = find_csv_files(input_dir)
    print(f"Found {len(input_files)} CSV files")
    
    # Step 2: First pass - Parse and standardize all files
    print("First pass: Parsing and standardizing files...")
    file_infos = [
        info for input_path in input_files
        if (info := parse_and_standardize_file(input_path)) is not None
    ]
    
    print(f"Processed {len(file_infos)} valid files")
    
    # Step 3: Second pass - Join ratings and dissipation data
    print("Second pass: Joining ratings and dissipation data...")
    processed_file_infos = process_series_tables_by_type(file_infos)
    
    # Step 4: Save processed data
    print("Saving processed data...")
    save_processed_files(processed_file_infos, output_dir)
    
    # Step 5: Generate reports
    print("Generating reports...")
    
    # Uncomment to generate header mapping report if needed
    # print("Generating header mapping report...")
    # generate_header_mapping_report(file_infos)
    
    print("Generating ESR coverage report...")
    generate_esr_coverage_report(output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
