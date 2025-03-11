import os
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
import logging
from series_report_generator import generate_all_reports
from thread_classes import TableType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    manufacturer: str
    relative_path: str
    raw_headers: List[str]      # Original headers before standardization
    mapped_headers: List[str]  # Headers after standardization
    file_type: TableType
    series_name: str
    filename: str  # The filename for output
    df: pd.DataFrame # The processed DataFrame


# Helper function to convert frequency string to numeric value in Hz
def freq_to_numeric(freq_str: str) -> Optional[int]:
    """Convert frequency string like "50Hz" or "10kHz" to numeric value in Hz"""
    try:
        # Extract numbers and units with regex
        if not (match := re.match(r'(\d+)\s*(k?hz+)', freq_str.lower())):
            return None
            
        value, unit = match.groups()
        # Convert to base Hz value
        if 'khz' in unit:
            return int(value) * 1000
        elif 'hz' in unit:
            return int(value)
        return None
    except (ValueError, TypeError):
        print(f"Error converting frequency to numeric: {freq_str}")
        return None

def numeric_to_freq_str(numeric_value: int) -> str:
    """Convert numeric value in Hz to frequency string like "50Hz" or "10kHz" """
    if numeric_value >= 1000:
        return f"{numeric_value // 1000}kHz"
    else:
        return f"{numeric_value}Hz"

 
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
    freq_match = re.search(r'(\d{1,})\s?(k|hz|khz)', header.lower())
    if freq_match:
        return f"{freq_match.group(1)}{'k' if 'k' in header else ''}Hz"
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

def map_header_to_standard_name(header: str) -> str:
    """
    Standardize a header using regex patterns.
    Returns the original header if no pattern matches.
    """
   
    
    patterns = {
        # Basic electrical parameters
        r".*?cap(?:acitance)?.*": "Capacitance", 
        r".*?esr.*": lambda x: f"ESR {extract_temp_freq(x)}".strip(),
        r".*?esl.*": lambda x: f"ESL {extract_temp_freq(x)}".strip(),
        r".*?(?:impedance|^z$).*": lambda x: f"Impedance {extract_temp_freq(x)}".strip(),
        r".*?max.*?ripple.*": lambda x: f"Max Ripple Current {extract_temp_freq(x)}".strip(),
        r".*?ripple.*|^rc": lambda x: f"Ripple Current {extract_temp_freq(x)}".strip(),
        r".*?surge.*": "Surge Voltage",
        r".*?(?:voltage|vdc|wv).*": "Voltage",
        r".*?leak.*current.*|^lc$": "Leakage Current",
        r".*?(?:tan.*[δd]|^df$|dissipation.*factor).*": "Dissipation Factor",
        r".*?(?:coefficient).*": "Coefficient",
        r"(cv|vc)\s*(\(µf\s*x\s*v\))?": "VC",

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
        r".*?(?:case|size).*(?:[φøo]?d?.*[×x].*l|[φøo]?d.*l|l.*[φøo]?d).*": "Case Size",
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
        r"^(?:[φøo]?d).*": "Case Size Diameter",\
        # Frequency pattern - matches standalone frequency values
        r"(?:freq\.?|frequency coefficient)?\s*(?:\(?hz\)?)?\s*(\d{1,3})\s*(k|khz|k hz|hz)\s*(?:to)?": lambda x: f"Frequency Coefficient {extract_frequency(x.strip())}",
        r"^≤?(?:\d+(?:\.\d+)?(?:\s*(?:hz|khz)))$": lambda x: f"Frequency Coefficient {extract_frequency(x.strip())}",
        r"^\d{1,3}\s*k$": lambda x: f"Frequency Coefficient {extract_frequency(x.strip())}",
        r"^\d{1,3}$": lambda x: f"Frequency Coefficient {x.strip()}Hz",
    }
    
    # Apply patterns to the header
    for pattern, replacement in patterns.items():
        if not re.match(pattern, str(header), flags=re.IGNORECASE):
            continue
        if callable(replacement):
            return str(replacement(header))
        else:
            return str(replacement)
    return header

def convert_dissipation_table_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a dissipation factor table with voltage columns into a two-column format.
    Input format:
        Voltage | 6.3V | 10V | 16V | ...
        Dissipation Factor | 0.12 | 0.14| 0.15| ...
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


def convert_frequency_table_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a frequency table with voltage columns into a two-column format.
    Input format:
        Frequency | Coefficient
        50Hz      | 0.12
        120Hz     | 0.14
        300Hz     | 0.15
    Output format:
        Frequency | 50Hz | 120Hz | 300Hz | ...
        Coefficient | 0.12 | 0.14| 0.15| ...
    """
    # Skip if already in correct format
    has_coefficient_column = re.search(r'coefficient', str(list(df.columns)[1]).lower())
    if not has_coefficient_column:
        return df
    
    # Set frequency values as index then transpose
    freq_df = df.set_index(df.columns[0])
    transposed_df = freq_df.T

    # Reset index and rename the index column to 'Frequency'
    transposed_df = transposed_df.reset_index().drop(['index'], axis=1)
    transposed_df.columns.name = None
    return transposed_df

def find_csv_files(input_dir: str) -> List[str]:
    """Find all CSV files in the input directory."""
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def remove_empty_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are all NaN or empty values."""
    df = input_df.copy()
    
    all_na_cols = [col for col in df.columns if df[col].isna().all()]
    all_empty_cols = [col for col in df.columns if (df[col].astype(str).str.strip() == '').all()]    
    cols_to_drop = list(set(all_na_cols + all_empty_cols))
    
    if cols_to_drop:
        logger.debug(f"Dropping {len(cols_to_drop)} empty columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df

def calculate_esr_from_dissipation(
    input_df: pd.DataFrame, 
    frequency: float = 120.0, 
    temperature: int = 20
) -> pd.DataFrame:
    """
    Compute ESR from Dissipation Factor and Capacitance at a specified frequency.
    Automatically applies frequency coefficient if a matching column exists.
    
    Args:
        input_df: DataFrame containing 'Dissipation Factor' and 'Capacitance' columns
        frequency: Frequency in Hz to calculate ESR at (default: 120Hz)
        temperature: Temperature in °C to include in the column name (default: 20)
        
    Returns:
        A new DataFrame with added ESR column
    """
    # Create a copy of the DataFrame to avoid modifying the input
    df = input_df.copy()
    # Find matching frequency coefficient column if it exists
    freq_str = f"{int(frequency)}Hz" if frequency < 1000 else f"{int(frequency/1000)}kHz"
    # Check if all required columns are present
    if not all(col in df.columns for col in ['Dissipation Factor', 'Capacitance']):
        return df

    def compute_esr_value(row):
        """
        Compute ESR from Dissipation Factor and Capacitance at a specified frequency.
        Formula: ESR = DF / (2π × f × C)
            - DF is Dissipation Factor (unitless)
            - f is frequency in Hz
            - C is capacitance in Farads
        Automatically applies frequency coefficient if available.
        """
        if not (pd.notnull(row['Dissipation Factor']) and pd.notnull(row['Capacitance']) and row['Capacitance'] > 0):
            return np.nan
        # used to account for the fact that the tanδ is higher at higher capacitance values
        capacitance_factor_offset = ((row['Capacitance']  - 1000) // 1000 + 1) * 0.02
        esr = (row['Dissipation Factor'] + capacitance_factor_offset) / (2 * 3.14159 * 120 * row['Capacitance'] * 1e-6)

        coef_column_name = f"Frequency Coefficient {freq_str}"

        return round(esr, 3)


    # Format temperature for column name
    temp_str = f"{temperature}°C"
    column_name = f"ESR(est.) {temp_str}@{freq_str}"
    
    df[column_name] = df.apply(compute_esr_value, axis=1)
    
    return df

def calculate_ripple_current_at_frequencies(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ripple current ratings at different frequencies using frequency coefficient data.
    Coefficients are normalized with 120Hz = 1.0.
    
    Args:
        input_df: DataFrame containing ripple current and frequency coefficient columns
        source_col: The source ripple current column to convert from (auto-detected if None)
        
    Returns:
        DataFrame with added ripple current columns for target frequencies
    """
    df = input_df.copy()
    
    def extract_frequency_columns(prefix: str) -> Dict[int, Dict[str, str]]:
        """Extract frequency information from columns with the given prefix"""
        result = {}
        
        for col in df.columns:
            if not col.startswith(prefix):
                continue
            if (freq_hz := freq_to_numeric(extract_frequency(col))) is None:
                continue
            result[freq_hz] = col
        
        return result
    
    ripple_cols = [col for col in df.columns if col.startswith('Ripple Current')]
    # Get the lowest temperature column
    source_col = sorted(ripple_cols, key=lambda x: int(extract_temp(x).replace('°C', '') or 0))[0]
    if not (source_freq_hz := freq_to_numeric(extract_frequency(source_col))):
        return df # No Ripple Current data available
    source_freq_str = numeric_to_freq_str(source_freq_hz)
    source_coef_col = f"Frequency Coefficient {source_freq_str}"
    
    # Extract frequency coefficient and ripple current columns
    if not (freq_coef_map := extract_frequency_columns('Frequency Coefficient')):
        return df  # No frequency coefficient data available

    # Exit if we couldn't find a valid source
    if source_freq_hz not in freq_coef_map:
        return df

    
    # Process each target frequency
    for target_freq_hz, target_coef_col in freq_coef_map.items():
        # Skip if target matches source
        if target_freq_hz == source_freq_hz:
            continue

        # Create output column name
        new_col_name = f"Ripple Current {extract_temp(source_col)}@{numeric_to_freq_str(target_freq_hz)}"
        
        # Skip if column already exists
        if new_col_name in df.columns:
            continue
        
        # Calculate new ripple current values
        def calculate_ripple(row):
            if pd.isna(row[source_col]) or pd.isna(row[target_coef_col]):
                return np.nan
            
            source_value = int(row[source_col])
            target_coef = float(row[target_coef_col])
            
            # If our source is 120Hz, we can directly apply the coefficient
            if source_freq_hz == 120:
                return round(source_value * target_coef, 2)
            
            # For any other source frequency, we need to use its coefficient to normalize
            # back to 120Hz equivalent first, then apply target coefficient
            if pd.isna(row.get(source_coef_col, None)) or row.get(source_coef_col, 0) == 0:
                return np.nan
                
            source_coef = row[source_coef_col]
            # Convert from source to 120Hz then to target
            return round((source_value / source_coef) * target_coef, 2)
        
        df[new_col_name] = df.apply(calculate_ripple, axis=1)
    
    return df

def extract_numerical_values_with_ranges(value: str) -> Union[str, Any]:
    """
    Extract numerical values from a string while preserving range indicators.
    Examples:
        "100V-200V" -> "100-200"
        "10 to 20 kHz" -> "10 to 20"
        "≥ 50Hz" -> "≥ 50"
        "10.5V" -> "10.5"
    
    Args:
        value: String value to process
        
    Returns:
        String with extracted numerical values and preserved range indicators
    """
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    if value.lower().strip() in ('nan', 'none', '', 'na', 'n/a'):
            return np.nan

    # Special case for "greater than" or "less than" indicators
    if any(indicator in value.lower() for indicator in ['≥', '>=', '>', '<', '≤', '<=']):
        # Extract the operator and the number
        op_match = re.search(r'([≥>≤<]=?)\s*(\d+(?:\.\d+)?)', value)
        if op_match:
            operator, number = op_match.groups()
            return f"{operator} {number}"
    
    # Check if this is a range with a separator (-, ~, ～, to)
    # Use regex to split by any range separator
    range_separator_pattern = r'[-~～]|\s+to\s*'
    if re.search(range_separator_pattern, value):
        parts = re.split(range_separator_pattern, value)
        
        # Extract numericals from each part
        numeric_parts = []
        for part in parts:
            num_match = re.search(r'(\d+(?:\.\d+)?)', part)
            if num_match:
                numeric_parts.append(num_match.group(1))
            else:
                numeric_parts.append('')
                

        # Get the actual separator used
        separator_match = re.search(range_separator_pattern, value)
        if separator_match:
            separator = separator_match.group(0)
            return separator.join(numeric_parts)
        else:
            # Fallback to hyphen if we can't determine the separator (shouldn't happen)
            return ' to '.join(numeric_parts)
    return value

def clean_and_convert_values(input_df: pd.DataFrame, cast_numeric_columns: bool = False) -> pd.DataFrame:
    """
    Standardize values in the dataframe based on column type.
    - Replace placeholder values (dashes, N/A, etc.) with None using regex
    - Remove commas and quotes from all columns
    - For numerical columns, extract numbers and convert to appropriate type
    - Apply specific standardization for known columns
    - Break Case Size columns into diameter (D) and length (L) components
    - Filter out rows that only have Voltage and Capacitance values but no other values
    
    Args:
        input_df: DataFrame to clean and convert
        
    Returns:
        A new DataFrame with standardized values
    """
    df = input_df.copy()
    
    numerical_columns = [
        'Voltage',
        'Capacitance', 
        'ESR', 
        'Impedance', 
        'Ripple Current',
        'Surge Voltage',
        'Leakage Current',
        'Dissipation Factor',
        'Frequency',
        'Case Size Diameter',
        'Case Size Length',
    ]
    
    placeholder_pattern = r'^(\s*|—|–|-|\.{1,3}|N/?A|n/?a|NA|na|nan)$'
    
    i = 0
    # While loop to handle the fact that we dynamically add columns as we process them
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
        if (col in numerical_columns) or (str(col).startswith('Frequency Coefficient')):
            df[col] = df[col].apply(extract_numerical_values_with_ranges)
            if cast_numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if str(col).startswith('Ripple Current'):
            df[col] = df[col].apply(extract_numerical_values_with_ranges)
            if cast_numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if max(df[col]) < 10:
                    # Rating is in A not mA
                    df[col] *= 1000
        i += 1
    
    # Filter out rows that only have Voltage and Capacitance values but no other values
    if 'Voltage' in df.columns and 'Capacitance' in df.columns:
        # Get all columns except Voltage and Capacitance
        other_cols = [col for col in df.columns if col not in ['Voltage', 'Capacitance']]
        
        if other_cols:
            # Keep rows where at least one other column has a non-NaN value
            has_other_data = df[other_cols].notna().any(axis=1)
            df = df[has_other_data | (~df['Voltage'].notna() & ~df['Capacitance'].notna())]
            
            # Log how many rows were filtered out
            removed_count = len(input_df) - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} rows with only Voltage and Capacitance values")
    
    return df

def make_column_names_unique(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names unique by appending numbers to duplicate names.
    For example, if there are two columns named "Voltage", the second one 
    will be renamed to "Voltage_1".
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
        FileInfo object containing the processed DataFrame and metadata, or None if file type can't be determined
    """
    manufacturer = os.path.basename(os.path.dirname(input_path))
    
    # Get filename and determine file type directly from the filename
    filename = os.path.basename(input_path)
    path = Path(filename)
    
    # Extract series name (everything before the first underscore)
    series_name = path.stem.split('_')[0].lower()
    
    # Determine file type from filename
    file_type = None
    if "dissipation" in path.stem.lower():
        file_type = TableType.DISSIPATION
    elif "frequency" in path.stem.lower():
        file_type = TableType.FREQUENCY
    elif "ratings" in path.stem.lower():
        file_type = TableType.RATINGS
    else:
        # Don't default to RATINGS, print a warning and return None
        logger.warning(f"Could not determine file type for {input_path}. Skipping file.")
        return None
    # Check if file is empty
    if os.path.getsize(input_path) == 0:
        logger.warning(f"File {input_path} is empty. Skipping.")
        return None
    
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='error')
    # Check if df has any data
    if df.empty:
        logger.warning(f"File {input_path} is empty. Skipping.")
        return None

    logger.debug(f"Processing file: {input_path}")
    logger.debug(f"Original columns: {list(df.columns)}")
    
    df = remove_empty_columns(df)
    raw_headers = list(df.columns)

    # If this is a dissipation factor table, pivot it
    if file_type == TableType.DISSIPATION:
        df = convert_dissipation_table_to_standard_format(df)

    if file_type == TableType.FREQUENCY:
        df = convert_frequency_table_to_standard_format(df)
        # Standard frequency values often used in datasheets
        standard_frequencies = ["50Hz", "60Hz", "120Hz", "300Hz", "1kHz", "10kHz", "100kHz"]

        # Expand columns with ranges
        df = expand_range_columns(df, standard_frequencies, value_converter=freq_to_numeric)

    df = df.rename(columns=lambda x: map_header_to_standard_name(x))
    logger.debug(f"After standardizing headers: {list(df.columns)}")
    
    # Make column names unique if there are duplicates
    df = make_column_names_unique(df)
    
    mapped_headers = list(df.columns)
    if series_name.lower() == 'rjf':
        pass
    df = clean_and_convert_values(df)
    logger.debug(f"After standardizing values: {list(df.columns)}")

    # Create file info with the DataFrame embedded
    file_info = FileInfo(
        manufacturer=manufacturer,
        relative_path=input_path,
        raw_headers=raw_headers,
        mapped_headers=mapped_headers,
        file_type=file_type,
        series_name=series_name,
        filename=filename,
        df=df
    )
    
    return file_info

def expand_range_columns(
    df: pd.DataFrame, 
    standard_values: Optional[List[str]] = None,
    value_converter: Optional[Callable[[str], Any]] = None
) -> pd.DataFrame:
    """
    Expand columns with ranges in their names into individual columns.
    For example:
        - "50Hz to 60Hz" becomes two columns: "50Hz" and "60Hz"
        - "10kHz or more" or "≥10kHz" gets mapped to all frequencies >= 10kHz in standard_values
    
    Args:
        df: DataFrame containing columns with ranges in their names
        standard_values: Optional list of standard values to use for "or more" patterns
        value_converter: Function to convert string values to comparable types for sorting.
                       If None, values are compared as strings.
    
    Returns:
        DataFrame with expanded columns
    """
    result_df = df.copy()
    
    # Use default string comparison if no converter provided
    if value_converter is None:
        value_converter = lambda x: x
    
    # Identify columns with ranges that need to be expanded
    for col in result_df.columns:
        col_str = str(col)
        
        # Case 1: "X to Y" pattern
        range_patterns = r'\s+to\s*|\s*,\s*|\s+•\s+'
        if re.search(range_patterns, col_str.lower()):
            range_parts = re.split(range_patterns, col_str.lower())
            # closed range
            if len(range_parts) == 2:
                start_value, end_value = range_parts
                result_df[start_value] = result_df[col]
                result_df[end_value] = result_df[col]
            # open ended range
            else:
                start_value = range_parts[0]
                result_df[start_value] = result_df[col]
            # Drop the original column
            result_df = result_df.drop(columns=[col])
      
        # Case 2: "X or more" pattern with standard values
        elif any(pattern in col_str.lower() for pattern in ["or more", "≥", ">="]) and standard_values:
            # Extract the base value using regex patterns for different notations            
            # Pattern 1: "X or more"
            if "or more" in col_str.lower():
                base_value = col_str.lower().split(" or more")[0].strip()
            # Pattern 2: "≥X" or ">=X"
            elif any(op in col_str for op in ["≥", ">="]):
                # Match pattern like "≥10kHz" or ">=10kHz"
                if match := re.search(r'[≥>][=]?\s*(\d+\s*[a-zA-Z]+)', col_str):
                    base_value = match.group(1)
            else:
                continue
                
            # Find all standard values that are greater than or equal to the base value
            for std_val in standard_values:
                # Convert values for comparison
                std_converted = value_converter(std_val)
                base_converted = value_converter(base_value)
                
                # Skip if conversion fails
                if std_converted is None or base_converted is None:
                    continue
                    
                # If standard value is >= base value, create a new column
                if base_converted <= std_converted:
                    result_df[std_val] = result_df[col]
            
            # Drop the original column
            result_df = result_df.drop(columns=[col])
        else:
            continue
    
    return result_df


def resolve_value_ranges(
    source_df: pd.DataFrame, 
    reference_df: pd.DataFrame, 
    column_name: str = 'Voltage'
) -> pd.DataFrame:
    """
    Expand ranges in a column by matching with actual values from a reference dataframe.
    Handles both closed ranges (e.g., "10-20") and open-ended ranges (e.g., "≥10", ">10").
    Special values like empty strings or "all" match all available values.
    
    Args:
        source_df: DataFrame containing ranges to expand
        reference_df: DataFrame containing specific values to match against
        column_name: Name of the column to process (default: 'Voltage')
        
    Returns:
        DataFrame with expanded rows where ranges are replaced with specific values
    """
    if column_name not in reference_df.columns:
        logger.warning(f"Reference dataframe does not contain column '{column_name}'")
        return source_df
    
    if column_name not in source_df.columns:
        logger.warning(f"Source dataframe does not contain column '{column_name}'")
        return source_df
    
    available_values = sorted(reference_df[column_name].unique())
    if not available_values:
        logger.warning(f"No values found in reference dataframe for column '{column_name}'")
        return source_df
    
    expanded_rows = []
    
    for _, row in source_df.iterrows():
        # Skip if the value is not a string and not NaN
        if not isinstance(row[column_name], str) and not pd.isna(row[column_name]):
            expanded_rows.append(row.to_dict())
            continue
            
        # Handle NaN, empty string, or 'all' as special cases that match all values
        if (pd.isna(row[column_name]) or 
            isinstance(row[column_name], str) and 
             (row[column_name].strip() == '' or 
              row[column_name].lower().strip() == 'all')):
            matching_values = available_values
        else:
            value_str = str(row[column_name])
                
            # Case 1: Check for closed range with separators
            separators = r'[~～\-/]|to'
            if re.search(separators, value_str):
                try:
                    start_str, end_str = re.split(separators, value_str)
                    start_num = float(start_str.strip()) if start_str.strip() else 0
                    end_num = float(end_str.strip()) if end_str.strip() else math.inf
                    
                    # Find matching values in the reference dataframe
                    matching_values = [v for v in available_values if start_num <= float(v) <= end_num]
                except (ValueError, TypeError):
                    # If we can't parse the range, keep the original row
                    logger.warning(f"Could not parse range '{value_str}' in column '{column_name}'")
                    expanded_rows.append(row.to_dict())
                    continue
            
            # Case 2: Check for open-ended range patterns (≥, >, ≤, <)
            elif re.search(r'[≥>≤<]=?', value_str):
                try:
                    # Extract the operator and the value
                    match = re.search(r'([≥>≤<]=?)\s*(\d+(?:\.\d+)?)', value_str)
                    if not match:
                        expanded_rows.append(row.to_dict())
                        continue
                        
                    operator, value = match.groups()
                    threshold = float(value.strip())
                    
                    # Find matching values based on the operator
                    if operator in ['≥', '>=']:
                        matching_values = [v for v in available_values if float(v) >= threshold]
                    elif operator == '>':
                        matching_values = [v for v in available_values if float(v) > threshold]
                    elif operator in ['≤', '<=']:
                        matching_values = [v for v in available_values if float(v) <= threshold]
                    elif operator == '<':
                        matching_values = [v for v in available_values if float(v) < threshold]
                    else:
                        matching_values = []
                except (ValueError, TypeError):
                    # If we can't parse the range, keep the original row
                    logger.warning(f"Could not parse open-ended range '{value_str}' in column '{column_name}'")
                    expanded_rows.append(row.to_dict())
                    continue
            else:
                # Not a range, keep the original row
                expanded_rows.append(row.to_dict())
                continue

        if matching_values:
            # Create a new row for each matching value
            for v in matching_values:
                new_row = row.to_dict()
                new_row[column_name] = v
                expanded_rows.append(new_row)
        else:
            # If no matches found, keep the original row
            expanded_rows.append(row.to_dict())

    return pd.DataFrame(expanded_rows)


def merge_ratings_with_dissipation(ratings_df: pd.DataFrame, dissipation_df: pd.DataFrame) -> pd.DataFrame:
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
    
    expanded_dissipation_df = resolve_value_ranges(dissipation_df, ratings_df, 'Voltage')

    expanded_dissipation_df = clean_and_convert_values(expanded_dissipation_df, cast_numeric_columns=True)
    ratings_df = clean_and_convert_values(ratings_df, cast_numeric_columns=True)

    # Left join to keep all ratings rows
    merged_df = pd.merge(
        ratings_df,
        expanded_dissipation_df[['Voltage', 'Dissipation Factor']],
        on='Voltage',
        how='left'
    )

    return merged_df

def merge_ratings_with_frequency(ratings_df: pd.DataFrame, frequency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a series that has both ratings and frequency data.
    Merges the frequency data into the ratings table and computes ESR.
    
    Args:
        ratings_df: DataFrame containing ratings data
        frequency_df: DataFrame containing frequency data
        
    Returns:
        DataFrame with merged data and computed ESR
    """
    # Create copies to avoid modifying the input DataFrames
    ratings_df = ratings_df.copy()
    ratings_df = clean_and_convert_values(ratings_df, cast_numeric_columns=True)
    ratings_df['VC'] = ratings_df['Capacitance'] * ratings_df['Voltage']
    cols_to_join = [col for col 
        in ['Capacitance', 'Voltage', 'VC', 'Case Size Diameter']
        if col in frequency_df.columns and col in ratings_df.columns
    ]
    expanded_frequency_df = frequency_df.copy()
    for col in cols_to_join:
        if not (col in frequency_df.columns and col in ratings_df.columns):
            continue
        expanded_frequency_df = resolve_value_ranges(expanded_frequency_df, ratings_df, col)

    cleaned_frequency_df = clean_and_convert_values(expanded_frequency_df, cast_numeric_columns=True)

    # Normalize frequency coefficients so 120Hz = 1.0
    frequency_columns = [col for col in cleaned_frequency_df.columns 
                         if re.match(r'^Frequency Coefficient \d+.*Hz$', col)]
    
    # Find the 120Hz column
    hz_120_column = next((col for col in frequency_columns 
                         if col == 'Frequency Coefficient 120Hz'), None)
    
    if hz_120_column:
        # Only normalize rows where 120Hz value is not null and not zero
        valid_rows = cleaned_frequency_df[hz_120_column].notna() & (cleaned_frequency_df[hz_120_column] != 0)
        unadjusted_df = cleaned_frequency_df.copy()
        if valid_rows.any():
            # For each frequency column, divide by the 120Hz value row-wise
            for freq_col in frequency_columns:
                cleaned_frequency_df.loc[valid_rows, freq_col] = (
                    unadjusted_df.loc[valid_rows, freq_col] / 
                    unadjusted_df.loc[valid_rows, hz_120_column]
                ).apply(lambda x: round(x, 2))
    
    # Ensure Capacitance is properly formatted for joining
    for col in cols_to_join:
        ratings_df[col] = ratings_df[col].astype(float)
        cleaned_frequency_df[col] = cleaned_frequency_df[col].astype(float)

    # Left join to keep all ratings rows
    merged_df = pd.merge(
        ratings_df,
        cleaned_frequency_df,
        on=cols_to_join if cols_to_join else None,
        how='left' if cols_to_join else 'cross'
    )
    return merged_df

def process_series_tables_by_type(file_infos: List[FileInfo]) -> List[FileInfo]:
    """Join dissipation data into ratings tables, standardize values,
    compute ESR, and reorder columns."""
    series_groups = defaultdict(list)
    for file_info in file_infos:
        series_groups[file_info.series_name].append(file_info)
    
    # Process each series
    for series_name, group in series_groups.items():
        ratings_info, dissipation_info, frequency_info = None, None, None

        for file_info in group:
            if file_info.file_type == TableType.RATINGS:
                ratings_info = file_info
            elif file_info.file_type == TableType.DISSIPATION:
                dissipation_info = file_info
            elif file_info.file_type == TableType.FREQUENCY:
                frequency_info = file_info

        # Process files based on what we have
        if ratings_info:
            if dissipation_info and "Dissipation Factor" not in ratings_info.df.columns:
                # We have both ratings and dissipation, and need to merge in dissipation data
                ratings_info.df = merge_ratings_with_dissipation(ratings_info.df, dissipation_info.df)
            if "Dissipation Factor" in ratings_info.df.columns:
                # Calculate ESR at standard frequency
                cleaned_df = clean_and_convert_values(ratings_info.df, cast_numeric_columns=True)
                ratings_info.df = calculate_esr_from_dissipation(cleaned_df, frequency=120.0)
            if frequency_info:
                if series_name.lower() == 'ky':
                    pass
                # Only ratings data, calculate ESR if dissipation factor is present
                ratings_info.df = merge_ratings_with_frequency(ratings_info.df, frequency_info.df)
                ratings_info.df = calculate_ripple_current_at_frequencies(ratings_info.df)
            if not dissipation_info and not frequency_info:
                # Only ratings data, calculate ESR if dissipation factor is present
                ratings_info.df = clean_and_convert_values(ratings_info.df, cast_numeric_columns=True)
    
    return file_infos


def save_processed_files(file_infos: List[FileInfo], output_dir: str) -> None:
    """
    Save processed files to the output directory and creates a consolidated CSV 
    with only priority columns from all series.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_saved = 0
    files_skipped = 0
    
    priority_cols = [
        'Voltage'
        , 'Capacitance'
        , 'ESR(est.) 20°C@20Hz'
        , 'Dissipation Factor'
        , 'ESR 20°C@100kHz'
        , 'Impedance 20°C@100kHz'
        , 'Case Size Diameter'
        , 'Case Size Length'
    ]

    all_series_data = []
    
    for file_info in file_infos:
        if file_info.file_type == TableType.DISSIPATION:
            files_skipped += 1
            logger.debug(f"Skipping dissipation file: {file_info.relative_path}")
            continue
        
        file_output_path = output_path / file_info.filename
        
        if file_info.file_type == TableType.FREQUENCY:
            # Save frequency coefficient files directly
            files_skipped += 1
            logger.debug(f"Skipping frequency coefficient file: {file_info.filename}")
            continue
            
        if file_info.file_type == TableType.RATINGS:
            df = file_info.df.copy()
            
            available_priority_cols = [col for col in priority_cols if col in df.columns]
            other_cols = sorted([col for col in df.columns if col not in priority_cols])
            df = df[available_priority_cols + other_cols]
            
            # Process numeric columns to remove trailing zeros
            for col in df.columns:
                # Convert to string, replace trailing zeros, and convert NaN to empty string
                df[col] = df[col].astype(str).replace(r'\.0$', '', regex=True).replace('nan', '')
            df = df.fillna('')

            # Save the individual file
            drop_cols = [c for c in df.columns if 'Frequency Coefficient' in c]
            df = df.drop(columns=drop_cols, errors='ignore')
            df.to_csv(file_output_path, index=False)
            files_saved += 1
            logger.debug(f"Saved {file_info.filename}")
            
            # Add metadata for consolidated file
            df['Series'] = file_info.series_name
            df['Manufacturer'] = file_info.manufacturer
            all_series_data.append(df)
    
    for df in all_series_data:
        # we need to choose one column per frequency and then remove the temps from the name
        ripple_cols = [c for c in df.columns if c.startswith('Ripple Current')]
        sorted_by_temp = sorted(ripple_cols, key=lambda x: int(extract_temp(x).replace('°C', '') or 0))
        col_name_maps = {}
        cols_to_drop = []
        for col in sorted_by_temp:
            # lower temp comes first, higher temp will be dropped
            new_col_name = f"Ripple Current @{extract_frequency(col)}"
            if new_col_name in col_name_maps:
                cols_to_drop.append(col)
            else:
                col_name_maps[new_col_name] = col
        # invert the mapping to map the new column names to the old column names
        df.rename(columns={v:k for k,v in col_name_maps.items()}, inplace=True)
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    consolidated_df = pd.concat(all_series_data, ignore_index=True)
    # Create a summary of which series have non-null values for each column
    series_with_data = {}
    for column in consolidated_df.columns:
        # Get series that have non-empty values in this column
        series_with_values = consolidated_df[consolidated_df[column].astype(str).str.strip() != '']['Series'].unique()
        if len(series_with_values) > 0:
            series_with_data[column] = sorted(series_with_values)
        
    # Create the merged ESR/Z column by coalescing ESR and Impedance
    consolidated_df['ESR/Z 20°C@100kHz'] = consolidated_df['ESR 20°C@100kHz'].combine_first(consolidated_df['Impedance 20°C@100kHz'])
    consolidated_df = consolidated_df.drop(columns=['ESR 20°C@100kHz', 'Impedance 20°C@100kHz'], errors='ignore')
    

    
    # Define columns for consolidated output
    output_cols = [
        'Series'
        , 'Manufacturer'
        , 'Capacitance'
        , 'Voltage'
        , 'ESR(est.) 20°C@120Hz'
        , 'Dissipation Factor'
        , 'ESR/Z 20°C@100kHz'
        , 'Ripple Current @120Hz'
        , 'Ripple Current @1kHz'
        , 'Ripple Current @10kHz'
        , 'Ripple Current @100kHz'
        , 'Case Size Diameter'
        , 'Case Size Length'
        ]
    
    # Keep only columns that exist in the dataframe
    available_cols = [col for col in output_cols if col in consolidated_df.columns]
    consolidated_df = consolidated_df[available_cols]
    
    # Ensure all NaN values are converted to empty strings
    consolidated_df = consolidated_df.fillna('')
    
    # Save the original consolidated data
    consolidated_output_path = output_path.parent / "all_series_priority_data.csv"
    consolidated_df = consolidated_df.drop_duplicates()
    consolidated_df.to_csv(consolidated_output_path, index=False)
    logger.info(f"Saved consolidated priority data to {consolidated_output_path}")
        
    logger.info(f"Saved {files_saved} files, skipped {files_skipped} files")

def main():
    """Main function that orchestrates the entire data processing pipeline."""
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
    
    # # Step 5: Generate reports
    # generate_all_reports(file_infos, output_dir, include_header_mapping=False)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
