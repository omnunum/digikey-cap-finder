"""
Report generation functions for series table processing.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Set
import pandas as pd
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_header_mapping_report(file_infos: List) -> None:
    """
    Print analysis of the processed files with header mappings.
    
    Args:
        file_infos: List of FileInfo objects
    """
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

def generate_column_coverage_report(
    output_base_dir: str, 
    column_pattern: str, 
    report_title: str,
    column_filter: Optional[Callable[[str], bool]] = None,
    specific_column: Optional[str] = None
) -> None:
    """
    Generic function to analyze processed files for column data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
        column_pattern: String pattern to match in column names (case-insensitive)
        report_title: Title for the report section
        column_filter: Optional function to further filter columns beyond the pattern match
        specific_column: Optional specific column to check for (e.g., 'ESR 20°C@120Hz')
    """
    print(f"\n{report_title}")
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
    files_with_matching_columns = 0
    files_with_specific_column = 0
    files_with_both = 0
    files_with_null_values = 0
    
    columns_by_file = {}
    null_value_counts = {}
    
    # Analyze each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            total_files += 1
            
            # Get all matching columns
            matching_cols = [
                col for col in df.columns 
                if column_pattern.lower() in col.lower() and 
                (column_filter is None or column_filter(col))
            ]
            
            has_matching_cols = len(matching_cols) > 0
            
            # Check for specific column if provided
            has_specific_col = specific_column in df.columns if specific_column else False
            
            # Update counters
            if has_matching_cols:
                files_with_matching_columns += 1
                columns_by_file[file_path.stem] = matching_cols
                
                # Count null values in matching columns
                for col in matching_cols:
                    null_count = df[col].isna().sum()
                    total_count = len(df)
                    if null_count > 0:
                        files_with_null_values += 1
                        null_value_counts[f"{file_path.stem}:{col}"] = (null_count, total_count)
                        break  # Count file only once if any column has nulls
            
            if specific_column:
                if has_specific_col:
                    files_with_specific_column += 1
                    
                    # Count null values in specific column
                    null_count = df[specific_column].isna().sum()
                    total_count = len(df)
                    if null_count > 0:
                        null_value_counts[f"{file_path.stem}:{specific_column}"] = (null_count, total_count)
                
                if has_matching_cols and has_specific_col:
                    files_with_both += 1
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Print summary statistics
    print(f"\nTotal files analyzed: {total_files}")
    print(f"Files with {column_pattern} data: {files_with_matching_columns} ({files_with_matching_columns/total_files*100:.1f}%)")
    
    if specific_column:
        print(f"Files with specific column '{specific_column}': {files_with_specific_column} ({files_with_specific_column/total_files*100:.1f}%)")
        print(f"Files with both general and specific columns: {files_with_both} ({files_with_both/total_files*100:.1f}%)")
    
    if files_with_matching_columns > 0:
        print(f"Files with null {column_pattern} values: {files_with_null_values} ({files_with_null_values/files_with_matching_columns*100:.1f}% of files with {column_pattern} data)")
    
    # Print details about files with null values
    if null_value_counts:
        print(f"\nFiles with null {column_pattern} values:")
        print("-" * 60)
        for file_col, (null_count, total_count) in sorted(null_value_counts.items(), key=lambda x: x[1][0]/x[1][1], reverse=True):
            print(f"{file_col}: {null_count}/{total_count} values null ({null_count/total_count*100:.1f}%)")
    
    # Print details about matching columns
    if columns_by_file:
        print(f"\nFiles with {column_pattern} columns:")
        print("-" * 60)
        for file_name, columns in sorted(columns_by_file.items()):
            print(f"{file_name}: {', '.join(columns)}")

def generate_esr_coverage_report(output_base_dir: str) -> None:
    """
    Analyze processed files to report on ESR data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
    """
    generate_column_coverage_report(
        output_base_dir=output_base_dir,
        column_pattern="ESR",
        report_title="ESR Data Analysis",
        specific_column='ESR 20°C@120Hz'
    )

def generate_ripple_coverage_report(output_base_dir: str) -> None:
    """
    Analyze processed files to report on Ripple Current data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
    """
    def is_max_ripple(col: str) -> bool:
        return 'max' not in col.lower()
    
    generate_column_coverage_report(
        output_base_dir=output_base_dir,
        column_pattern="ripple",
        report_title="Ripple Current Data Analysis",
        column_filter=is_max_ripple
    )
    
    # Also generate a report specifically for max ripple current
    generate_column_coverage_report(
        output_base_dir=output_base_dir,
        column_pattern="max ripple",
        report_title="Max Ripple Current Data Analysis"
    )

def generate_impedance_coverage_report(output_base_dir: str) -> None:
    """
    Analyze processed files to report on Impedance data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
    """
    generate_column_coverage_report(
        output_base_dir=output_base_dir,
        column_pattern="impedance",
        report_title="Impedance Data Analysis",
        specific_column='Impedance 20°C@100kHz'
    )

def generate_all_reports(file_infos: List, output_dir: str, include_header_mapping: bool = False) -> None:
    """
    Generate all reports in one function call.
    
    Args:
        file_infos: List of FileInfo objects
        output_dir: Directory containing processed files
        include_header_mapping: Whether to include header mapping report
    """
    print("\nGenerating reports...")
    
    if include_header_mapping:
        print("Generating header mapping report...")
        generate_header_mapping_report(file_infos)
    
    print("Generating ESR coverage report...")
    generate_esr_coverage_report(output_dir)
    
    print("Generating Ripple Current coverage report...")
    generate_ripple_coverage_report(output_dir)
    
    print("Generating Impedance coverage report...")
    generate_impedance_coverage_report(output_dir) 