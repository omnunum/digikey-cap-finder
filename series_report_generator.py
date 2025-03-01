"""
Report generation functions for series table processing.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
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

def generate_ripple_coverage_report(output_base_dir: str) -> None:
    """
    Analyze processed files to report on Ripple Current data availability and quality.
    
    Args:
        output_base_dir: Base directory for processed files
    """
    print("\nRipple Current Data Analysis")
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
    files_with_ripple_current = 0
    files_with_max_ripple = 0
    files_with_both = 0
    files_with_null_ripple = 0
    
    ripple_columns_by_file = {}
    null_ripple_counts = {}
    
    # Analyze each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            total_files += 1
            
            # Check for ripple current columns
            ripple_cols = [col for col in df.columns if 'ripple' in col.lower()]
            has_ripple = len(ripple_cols) > 0
            
            # Check for max ripple columns specifically
            max_ripple_cols = [col for col in ripple_cols if 'max' in col.lower()]
            has_max_ripple = len(max_ripple_cols) > 0
            
            # Update counters
            if has_ripple:
                files_with_ripple_current += 1
                ripple_columns_by_file[file_path.stem] = ripple_cols
                
                # Count null values in ripple current columns
                for col in ripple_cols:
                    null_count = df[col].isna().sum()
                    total_count = len(df)
                    if null_count > 0:
                        files_with_null_ripple += 1
                        null_ripple_counts[f"{file_path.stem}:{col}"] = (null_count, total_count)
                        break  # Count file only once if any ripple column has nulls
            
            if has_max_ripple:
                files_with_max_ripple += 1
            
            if has_ripple and has_max_ripple:
                files_with_both += 1
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Print summary statistics
    print(f"\nTotal files analyzed: {total_files}")
    print(f"Files with ripple current data: {files_with_ripple_current} ({files_with_ripple_current/total_files*100:.1f}%)")
    print(f"Files with max ripple current data: {files_with_max_ripple} ({files_with_max_ripple/total_files*100:.1f}%)")
    print(f"Files with both regular and max ripple: {files_with_both} ({files_with_both/total_files*100:.1f}%)")
    print(f"Files with null ripple current values: {files_with_null_ripple} ({files_with_null_ripple/files_with_ripple_current*100:.1f}% of files with ripple data)")
    
    # Print details about files with null ripple values
    if null_ripple_counts:
        print("\nFiles with null ripple current values:")
        print("-" * 60)
        for file_col, (null_count, total_count) in sorted(null_ripple_counts.items(), key=lambda x: x[1][0]/x[1][1], reverse=True):
            print(f"{file_col}: {null_count}/{total_count} values null ({null_count/total_count*100:.1f}%)")
    
    # Print details about ripple current columns
    if ripple_columns_by_file:
        print("\nFiles with ripple current columns:")
        print("-" * 60)
        for file_name, columns in sorted(ripple_columns_by_file.items()):
            print(f"{file_name}: {', '.join(columns)}")

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