"""
PDF table extraction using Claude 3.5 Sonnet with direct PDF handling.
"""
import os
import io
import re

from collections import defaultdict
from typing import Dict, List, Union, Any, Optional
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import pdfplumber

from thread_classes import TableType, ExtractTableThread, CleanTableThread, extract_configuration, clean_configuration


@dataclass
class FileInfo:
    """Information about a processed file, including data and metadata."""
    dataframe: pd.DataFrame
    file_type: str
    file_path: Path


def save_dataframes(data: Dict[str, pd.DataFrame], output_path: Path, table_type: TableType) -> None:
    """Save extracted DataFrames to CSV files using standardized naming based on TableType."""
    if not data:
        print(f"No data to save for {table_type.value}")
        return
    
    def get_extended_path(table_name: str) -> Path:
        """Create sanitized path for a table with extended naming."""
        safe_name = re.sub(r'[^\w\s-]', '', table_name).replace(' ', '-')
        return output_path.with_name(f"{output_path.stem}_{table_type.value}_{safe_name}.csv")
    
    def save_extended_table(table_name: str, df: pd.DataFrame) -> None:
        """Save a table with extended naming."""
        path = get_extended_path(table_name)
        df.to_csv(path, index=False)
        print(f"Saved table '{table_name}' ({df.shape[0]}×{df.shape[1]}) to {path}")
        
    # Initialize with the first table
    items = list(data.items())
    largest_table_name, largest_df = items[0]
    largest_size = largest_df.shape[0] * largest_df.shape[1]
    
    # Process all tables in a single loop
    for name, df in items[1:]:  # Skip the first item as we've already processed it
        # Calculate table size (rows × columns)
        size = df.shape[0] * df.shape[1]
        
        if size > largest_size:
            # Found a new largest table - save previous largest with extended name
            save_extended_table(largest_table_name, data[largest_table_name])
            
            # Update largest table
            largest_table_name = name
            largest_size = size
        else:
            # Not the largest table, save with extended name
            save_extended_table(name, df)
    
    # After processing all tables, save the largest one with the standard name
    standard_csv_path = output_path.with_name(f"{output_path.stem}_{table_type.value}.csv")
    data[largest_table_name].to_csv(standard_csv_path, index=False)
    print(f"Saved primary {table_type.value} table '{largest_table_name}' ({data[largest_table_name].shape[0]}×{data[largest_table_name].shape[1]}) to {standard_csv_path}")
    
    # Print summary if multiple tables were found
    if len(data) > 1:
        print(f"Processed {len(data)} tables for {table_type.value}. Largest table was '{largest_table_name}'.")


def parse_xml_tables(text: str) -> Dict[str, pd.DataFrame]:
    """Parse tables with XML tags and csv format."""
    tables = {}
    
    # Find all table sections
    sections = re.finditer(r'<title>(.*?)</title>\s*<data>(.*?)</data>', text, re.DOTALL)
    
    for section in sections:
        title = section.group(1).strip()
        table_text = section.group(2).strip()
        
        # Collect problematic rows
        problematic_rows: dict[str, list[str]] = defaultdict(list)

        # Try to parse the entire table with error handling
        try:
            df = pd.read_csv(
                io.StringIO(table_text),
                on_bad_lines=lambda x: problematic_rows[title].append(str(x)),
                engine="python",
                na_filter=True,
                skip_blank_lines=True
            )
            if title not in tables:
                tables[title] = df
            else:
                tables[title] = pd.concat([tables[title], df], ignore_index=True)
        except pd.errors.ParserError as e:
            print(f"\nError parsing table '{title}':")
            print(f"Error: {str(e)}")
        
        # Print collected problematic rows if any
        for title, rows in problematic_rows.items():
            print(f"\nProblematic rows in table '{title}':")
            if title in tables:
                print(f"Header: {','.join(tables[title].columns)}")
            print("Rows:")
            for row in rows:
                print(f"  {row}")
            print()
    
    return tables


def main():
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Create output directory
    base_output_dir = Path("series_tables")
    
    for manufacturer, config in extract_configuration.items():
        # Process each PDF in series_pdfs directory
        pdf_dir = Path(f"series_pdfs/{manufacturer}")
        for pdf_path in pdf_dir.glob("*.pdf"):
            # Create output directory
            output_dir = base_output_dir / manufacturer
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / pdf_path.stem
            
            # Process each prompt config
            for prompt_config in config.prompt_configs:
                table_type_enum = prompt_config.table_type
                prompt = prompt_config.prompt
                
                # Check if any file with this prefix exists (including marker files)
                prefix = f"{pdf_path.stem}_{table_type_enum.value}"
                matching_files = list(output_dir.glob(f"{prefix}*.csv"))
                
                if matching_files:
                    file_list = ", ".join(f.name for f in matching_files)
                    print(f"\nSkipping {table_type_enum.value} extraction for {pdf_path.name} - found: {file_list}")
                    continue
                    
                print(f"\nProcessing {pdf_path.name} for {table_type_enum.value}")
                
                # Create and run a table extraction thread for the specific prompt
                thread = ExtractTableThread(
                    api_key=api_key,
                    pdf_path=pdf_path,
                    user_prompt=prompt,
                    name=f"{manufacturer}_{table_type_enum.value}",
                    max_tokens=8096
                )
                if not all(re.search(pattern, thread.load_pdf_text(pdf_path)) for pattern in prompt_config.required_patterns):
                    print(f"Skipping {pdf_path.name} for {table_type_enum.value}"
                        , " - required patterns not found: {', '.join(prompt_config.required_patterns)}")
                    continue
                
                # Extract tables
                response_text = thread.execute()
                print(f"Extracted response text length: {len(response_text)}")
                
                # Parse tables from response
                tables = parse_xml_tables(response_text)
                if not tables:
                    print(f"No tables found for {pdf_path.name} with prompt {table_type_enum.value}")
                    # Create empty marker file
                    not_found_path = output_dir / f"{prefix}_(not found).csv"
                    not_found_path.touch()
                    print(f"Created marker file: {not_found_path}")
                    continue
                print(f"Extracted {len(tables)} table entries")
                
                # Skip the rest if we don't need to clean the tables
                clean_config = clean_configuration.get(manufacturer)
                
                # Find matching clean prompt config
                clean_prompt_config = None
                if clean_config:
                    for config in clean_config.prompt_configs:
                        if config.table_type == table_type_enum:
                            clean_prompt_config = config
                            break
                
                if not clean_prompt_config:
                    save_dataframes(tables, output_path, table_type_enum)
                    continue
        
                cleaned_tables = {}
                for table_name, table_data in tables.items():
                    # Clean tables
                    raw_response = CleanTableThread(
                        api_key=api_key,
                        max_tokens=1024,
                        name=f"{manufacturer}_{table_type_enum.value}_{table_name}",
                        table_data=table_data.to_csv(index=False),
                        user_prompt=clean_prompt_config.prompt,
                    ).execute()
                    
                    try:
                        match = re.search(r"(?<=<csv>)[\s\S]*?(?=</csv>)", raw_response)
                        if match:
                            cleaned_table_csv = match.group(0)
                            cleaned_tables[table_name] = pd.read_csv(io.StringIO(cleaned_table_csv))
                        else:
                            print(f"Warning: No CSV data found in cleaning response for {table_name}")
                            cleaned_tables[table_name] = table_data
                    except Exception as e:
                        print(f"Error cleaning table {table_name}: {e}")
                        cleaned_tables[table_name] = table_data
                # Save this specific table type
                save_dataframes(cleaned_tables, output_path, table_type_enum)
                    
if __name__ == "__main__":
    main()
