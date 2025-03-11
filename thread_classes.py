"""
Thread classes for PDF table extraction and cleaning using Claude.
"""
import os
import base64
import re
from typing import Dict, Generator, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from anthropic.types import TextBlockParam, DocumentBlockParam, Base64PDFSourceParam, CacheControlEphemeralParam
from anthropic_api import ModelType, Thread, BlockParams


class TableType(Enum):
    FREQUENCY = "Frequency Coefficient"
    DISSIPATION = "Dissipation Factor"
    RATINGS = "Standard Ratings"


@dataclass
class ExtractTableThread(Thread):
    """Thread for extracting tables from PDFs using Claude."""
    
    pdf_path: Path
    system_prompt: str = """
Directive:
    - You are an AI model specifically designed to extract tables from PDFs efficiently and completely by responding with tagged data.

Critical Requirements:
    - Each table must be tagged: <title>{name}</title> and <data>{data}</data>
        - This is an absolute requirement and any deviation from this will result in a complete failure.
    - Start the response by stating the table name you are extracting and how you expect to identify it.
    - Never repeat values you have already extracted. Always extract new values.
    - While adhering to the previous uniqueness rule, maximize your response size by extracting as much as possible from the document in each response.
    - When a table spans multiple pages, extract ALL rows before moving to the next table
    - Output <end_of_document> when you have reached the end of all tables. Do not use any other way to communicate you have extracted all data.
    - Do not hallucinate tables that don't exist, read until the end of the document and then just stop.
    - Never summarize or aggregate, extract exact values only
    - If there are two tables with the same name, output each data as a separate table with the same title.
        - Make sure not to combine the tables in the output in case they have different column names/amount of columns.

Response Formatting Details:
    - Do not change the table title based on the voltage (like "Capacitor Characteristics 16V") or whether or not its a continuation (like "Capacitor Characteristics (Continued)").  
    - Use CSV format for the data
    - Quote all names of the columns
    - Include all columns in the table *in the order they are found in the input document*

How to Read from Tables Correctly:
    - When column names are formatted using vertically nested title cells, combine the titles. 
        - The upper cells will have the general name of the metric and the lowest/deepest level will usually have the units.
    - For values spanning multiple rows, broadcast them to all applicable rows when flattening
    - As you extract the table in parts across multiple pages, make sure the columns are all in the same order.
    - Include all columns and values from the input tables in the output.
    - Be careful to make sure that the amount of values and column names match, there should not be any fully empty columns and there should not be more columns than values.  
    - There are not multiple columns with the same name, if it looks like there are its just the same column name in a different group of columns with different values. 
        - We want to combine these columns from different groups into the output "vertically" using the same column names.  The resulting table should be much "taller" than the input.
    - Some of the specified columns may not be present in all tables, in that case do not specify them in the output.
    - Pay attention to details specified in foot notes or in relevant supplementary tables, and include these details in the column names.  
        - Examples are temp and frequency of ratings.

Additional Instructions:
    - Never ask for permission to continue, keep extracting until you reach the document end
    - If you reach a response limit mid-table, complete the current row and indicate table continuation
    - Process tables sequentially but maximize data extraction in each response
"""
    frequency_coefficient_prompt = """
Please extract the following tables:
- Required: "Frequency coefficient" 
    - Required columns (always present):
        - Capacitance (µF) 
        - Voltage (Vdc)
        - CV (Capacitance * Voltage)
        - {frequency}
            - Options like (50,120,300,1k,10k,100k or more) with or without the "Hz" units in the input
        - Coefficient
Do not output any other tables.
Notes about the "Frequency coefficient" table:
    - CRITICAL: There will never be a dedicated "Frequency" column AND individual {frequency} columns in the same table
        - If you see {frequency} columns, DO NOT output a "Frequency" column.  The standalone "Frequency" column is rare.
        - When you see "Frequency(Hz)" in the same header box as "Capacitance(µF)" or "Voltage(Vdc)", it is just a label to designate the numbers to the right of it as frequency headers.
            - It does not mean there is a "Frequency" column in the table.
    - {frequency} columns and the Coefficient values are always in the table
    - The {frequency} values are the headers above each column
        - Remove the "or more" from the frequency values, standarize to ">= {value}"
        - Ensure the units of the frequency values (Hz/kHz) are present in the output even if they are not in the input
    - CV values must go into their own column in the output, they cannot share a column with Voltage or Capacitance
    - Sometimes the input table is called "RATED RIPPLE CURRENT MULTIPLIERS"
    - Name the table "Frequency Coefficient" in the output
    - Sometimes the table has a rating above it like "(25 to 100Vdc)"
        - Use this rating as values in the related columns (Voltage in this case)
    - If the table looks like a graph/chart instead of a table, ignore it for now and do not include anything in the output
    - My use of `{frequency}` implies that there will be a column for each frequency value
    - Always have all required columns in the output even if there are no values for them
    - Keep the "{value} to {value}" format for all values if it is present
        - if there is another character like "{value} ~ {value} " or "{value} - {value}" between the values replace it with " to "
    - If voltage or capacitance is present and exits across multiple rows, broadcast the values to all relevant rows
Examples to Avoid:
    Pivoted Table:
    ```csv
Rated capacitance (µF),Coefficient
50Hz to 60Hz,0.7
120Hz,0.75
1kHz,0.9
10kHz to 100kHz,1
    ```
    Explanation:
        - Notice that while the capacitance header is present, the values are actually the frequency values
        - The frequency values should be columns instead of row values
        - The coefficient is present but the capacitance values are missing
    Correct Output:
    ```csv
Rated capacitance (µF), 50Hz to 60Hz, 120Hz, 1kHz, 10kHz to 100kHz
470 to 1000,0.7, 0.75, 0.9, 1
    ```
"""
    # Configuration for different manufacturers
    configuration = {
        "nichicon": {
            "model": ModelType.SONNET,
            "prompts": {
                # ratings are generated from series_info_extractor_nichicon.py
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        },
        "chemi-con": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.DISSIPATION: """
Please extract the following table:
- Optional: "Dissipation Factor (tan δ)" (on the first page if it exists)
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
Only output a table for the dissipation factor when you find a discrete table for it.  Do not include data from the ratings/characteristics table.
Do not output any other tables.
""",
                TableType.RATINGS: """
Please extract the following table:
- Required: "STANDARD RATINGS" (on subsequent pages), possible columns include (in no particular order and not exhaustive)
    - WV (Vdc) 
    - Cap (µF) 
    - Case size φD×L(mm) 
    - tanδ 
    - Rated ripple current (mArms/ {temp}, {frequency}) 
    - Part No.
    - ESR (Ω max./{frequency}) {temp}
Do not output any other tables.

Notes about "STANDARD RATINGS" table:
    - For some of the ratings, there are notes about the temp and frequency earlier in the document.  Please take those notes and integrate them into the relevant column names.
        - For instance, the temp and frequency of the tan value is noted earlier in the document in the dissipation factor table.
""", 
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        }, 
        "samsung": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.DISSIPATION: """
Please extract the following table:
- Optional: "Dissipation Factor (tan δ)" (on the first page if it exists)
    - Required columns (always present):
        - tanδ {temp} {frequency}
        - Rated Voltage (Vdc) 
Only output a table for the dissipation factor when you find a discrete table for it.  Do not include data from the ratings/characteristics table.
Do not output any other tables.
""",
                TableType.RATINGS: """
Please extract the following table:
- Required: "STANDARD RATINGS" (on subsequent pages), possible columns include (in no particular order and not exhaustive)
    - Rated Voltage (Vdc) 
        - Taken from the WV column from both input tables
        - Its the header at the top of the tables going horizontally across the top
        - Ignore the code after the voltage value
    - Rated Capacitance (µF) 
        - Taken from leftmost column from both input tables
        - Its the first column to the left of the "Case size" table, treat this like another header column but vertical
        - Ignore the code after the capacitance value
    - Case size φD×L(mm) 
        - The values in the cells of the Case size table
    - Rated ripple current (mArms/ {temp}, {frequency}) 
        - Taken from the values in the "Maximum permissible ripple current" table
        - Match to case size data using the voltage and capacitance values
Do not output any other tables.

Notes about "STANDARD RATINGS" table:
    - We want the input tables joined together to form a single table, so make sure to join the values in the tables by voltage value (not code) and capacitance value (not code)
    - Most rows will have all of the columns filled, but some rows may be missing values.
    - For some of the ratings, there are notes about the temp and frequency earlier in the document.  Please take those notes and integrate them into the relevant column names.
        - For instance, the temp and frequency of the tan value is noted earlier in the document in the dissipation factor table.
    - Pay close attention to which rating values belong to which voltage and capacitance values.
        - Some ratings will not exist at a given voltage and capacitance, and that is okay, ignore them.
        - Take extra time to get the first value of each voltage correct so that it has the correct capacitance.
""", 
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        },
        "kemet": {
            "model": ModelType.SONNET,
            # kemet does not have a frequency coefficient table in any of the sheets
            "prompts": {
                TableType.RATINGS: """
Please extract the following table:
- Required: "Table 1" (and all continuation tables)
    - possible columns include (in no particular order and not exhaustive)
        - Rated Voltage
        - Surge Voltage
        - Rated Capacitance
        - Case Size
        - DF
        - RC
        - Z
        - ESR
        - LC
        - Part Number
Do not output any other tables.

Notes about "Table 1":
    - Name the table "Standard Ratings" in the output.
    - Do not extract any other tables.
    - Do not specify if the table is continued in your response, just name them all "Standard Ratings".
    - As you extract the table in parts across multiple pages, make sure they are all in the same order.
    - Include all columns and values from the input tables in the output.
    - Any column names in the input tables that are not specified above should be taken as-is.
    - Be careful to make sure that the amount of values and column names match, there should not be any empty columns.  
    - Some of the specified columns may not be present in all tables, in that case do not specify them in the output.
"""
            }
        }, 
        "elna": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.DISSIPATION: """
Please extract the following table:
- Optional: "Tangent of loss angle (tan δ)"
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
Only output a table for the dissipation factor when you find a discrete table for it.  Do not include data from the standard ratings table.
Do not output any other tables.
""",
                TableType.RATINGS: """
Please extract the following table:
- Required: "Standard Ratings" 
    - Required columns (always present): 
        - Part No.
        - Rated voltage (V)
        - Rated capacitance (μF)
        - Case øD x L
    - Optional columns (non exhaustive): 
        - Size Code,ESR(Ω max.) {temp}
        - Rated ripple current (mArms) {frequency}
        - (tan δ) {temp} {frequency}
Do not output any other tables.

Notes about "Standard Ratings" table new table formats: 
    - This style of table can be determined by the line on the first page "CAT.No.2023/2024E" (or some other recent year)
    - Importantly, Rated Voltage looks like a column header at the top of the table but is actually a value like "25 (1T)" we dont want the voltage code (1T), we just the value 25.  
    - Each voltage rating applies to a group of columns.  Each group has the same columns, and we want all of the data for those columns to be combined across groups.
    - Rated Capacitance is on the left side of the table and it covers all groups of columns for those rows.

Notes about "Standard Ratings" table old table formats:
    - This style of table can be determined ty the line on the first page "This catalog printed in U.S.A. on 1/2001." 
        - The exact year will not be the same but it will be much older than a few years ago.
    - The voltage colun will come from the rows in the table that only contain the voltage value and none of the other column values.
        - Take the voltage value and apply it to subsequent rows until we see a new voltage value.

For all formats:
    - Below the Standard Ratings table, there are notes about the temp and frequency of some of the ratings.  
        - Please take those notes and integrate them into the relevant column names.
    - Additionally, rating context can be found in tables that come before the Standard Ratings table (like with the tan value temp and frequency)
""", 
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        },
        "panasonic": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.DISSIPATION: """
Please extract the following table:
- Optional: "Dissipation Factor (MAX)" 
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
Only output a table for the dissipation factor when you find a discrete table for it.  Do not include data from the characteristics list table.
Do not output any other tables.
""",
                TableType.RATINGS: """
Please extract the following table:
- Required: "Characteristics list" 
    - Required columns (always present): 
        - Rated voltage (V)
        - Rated capacitance (μF)
        - Case Size (mm) øD 
        - Case Size (mm) L
        - Part No.
    - Optional columns (non exhaustive):
        - Ripple current (mArms) {temp} {frequency}
        - ESR {temp} {frequency} 
        - tan δ {temp} {frequency}
        - Endurance (h) {temp}
        - Impedance {temp} {frequency}
        - Lead diameter (ød)
        - Lead space straight
        - Lead space Taping ✽B
        - Lead space Taping ✽H
        - Min packaging qty Straight leads
        - Min packaging qty Taping
For all formats:
    - Below the Characteristics list table, there are notes about the temp and frequency of some of the ratings.  
        - Please take those notes and integrate them into the relevant column names.
    - Additionally, rating context can be found in tables that come before the Characteristics list table (like with the tan value temp and frequency)
Do not output any other tables.
""", 
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        },
        "rubycon": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.DISSIPATION: """
Please extract the following table:
- Optional: "Dissipation Factor (MAX)" 
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
Only output a table for the dissipation factor when you find a discrete table for it.  Do not include data from the standard size table.
Do not output any other tables.
""",
                TableType.RATINGS: """
Please extract the following table:
- Required: "STANDARD SIZE" 
    - Required columns (always present): 
        - Rated voltage (V)
        - Rated capacitance (μF)
        - Size øD x L
    - Optional columns (non exhaustive):
        - Rated ripple current (mArms) {temp} {frequency}
        - ESR {temp} {frequency} 
        - Impedance {temp} {frequency}
Do not output any other tables.

Notes about the "STANDARD SIZE" table:
    - Around the Standard Ratings table, there are notes about the temp and frequency of some of the ratings.  
        - Please take those notes and integrate them into the relevant column names.
    - Additionally, rating context can be found in the SPECIFICATIONS table (like with the tan value temp and frequency)
""", 
                TableType.FREQUENCY: frequency_coefficient_prompt
            }
        },
    }
    
    def execute(self) -> str:
        """
        Extract tables from the PDF.
        
        Processes the conversation thread with Claude, handling the continuation
        logic specific to table extraction.
        
        Returns:
            Full text of all responses containing table data
        """
        # Prepare initial content based on model type
        initial_content: BlockParams
        if self.model == ModelType.SONNET:
            initial_content = [DocumentBlockParam(
                type="document",
                source=Base64PDFSourceParam(
                    type="base64",
                    media_type="application/pdf",
                    data=self.load_pdf_data(self.pdf_path)
                ),
                cache_control=CacheControlEphemeralParam(
                    type="ephemeral"
                )
            )]
        else:
            initial_content = [TextBlockParam(
                type="text",
                text=self.load_pdf_text(self.pdf_path),
                cache_control=CacheControlEphemeralParam(
                    type="ephemeral"
                )
            )]
        
        complete_response_text = []
        
        # Start the conversation and get responses
        thread_iter = self.iterate_thread(initial_content)
        response_text = next(thread_iter)
        
        # Process responses until end of document
        while True:
            # Check for end of document marker
            if "<end_of_document>" in response_text:
                # End of document reached, clean response and finish
                response_text = response_text.replace("<end_of_document>", "").strip()
                complete_response_text.append(response_text)
                break
            
            # Save this response
            complete_response_text.append(response_text)
            
            # Send "continue" and get next response
            try:
                response_text = thread_iter.send("<continue>")
            except StopIteration:
                break
        
        return "\n".join(complete_response_text)


@dataclass
class CleanTableThread(Thread):
    """Thread for cleaning and transforming extracted table data using Claude."""
    
    table_data: str
    system_prompt: str = """
Directive:
    - You are an AI model specifically designed to clean and transform CSV table data.
    - You will receive one table at a time in CSV format.
    - You are going to be recieving CSV table that was output by an AI model and thus may contain errors.
    - Your job is to clean the data and make sure it is valid.
Critical Requirements:
    - Tag all csv data with <csv>{data}</csv> tags.
    - Make sure to thorougly think through what you need to do before responding, but once you enter the <csv> tag you will only respond with the cleaned data and nothing else until the tag is closed.
    - Follow all user instructions exactly.
    - Ensure that the values are not altered unless explicitly instructed to do so.
    - Any values or headers in the input that have double quotes should continue to have them in the output

How to clean the CSV data:
    - Sometimes data gets misattributed to the wrong column.  You may be asked to remove a header but preserve the values under it.
    - Sometimes there will be extra column headers that you need to remove.
        - Note that you may need to remove a haeder from one column (shifting all headers to the left) and then a column from the right (which would have empty values)
"""
    primary_table_prompt = """
Instructions:
    - The columns that are numbers are actually frequency values, ensure that they have the correct units Hz/kHz
    - If there is a header that is "Frequency" and there are also other individual frequency headers, remove the "Frequency" header (but not the values under it).
        - Do not remove the values under the other Frequency header (we want to keep that column), just the "Frequency" header itself.
    - Ensure that outside of removing headers the order of the columns is the same as the input.
    - Drop any dangling empty columns (via comma removal) if any headers were removed to make the CSV continue to be valid
        - This will ensure that after header removal the number of columns and values match up.
    - Replace ratings that have phrases like "or less" or "or more" with the relevant operator (<, ≤, >, ≥)
        - Example: "56 or less" -> "≤56"
        - Example: "10 k to" -> "≥10kHz"
    - Always put the operator before the value, even if that means you have to flip the operator to do so
        - Example: "1000<" -> ">1000"
    - Replace phrases like "All CV value" with just "all"
    - Replace ranges in headers and values that are not in the format "a to b" or "a-b" with "a to b"
        - Examples: 
            - "10k・100k Hz" -> "10kHz to 100kHz"
            - "50-60Hz" -> "50Hz to 60Hz"
        - Use any separators you see fit like "-", "・", "~", to identify ranges that need to be standardized.
    - Remove units from values as long as they are in the header
Examples to Avoid:
    Input:
        Frequency(Hz),Capacitance(µF),120,1k,10k,100k
        27 to 180,0.4,0.75,0.9,1.0,

    Output:
        Capacitance(µF),27 to 180,120 Hz,1 kHz,10 kHz,100 kHz
        0.4,0.75,0.9,1.0,1.1,1.2

Examples to Follow:
    Input:
        Frequency(Hz),Capacitance(µF),120,1k,10k,100k
        27 to 180,0.4,0.75,0.9,1.0,

    Output:
        Capacitance(µF),120Hz,1kHz,10kHz,100kHz
        27 to 180,0.4,0.75,0.9,1.0
"""
    # Configuration for different manufacturers
    configuration = {
        "nichicon": {
            "model": ModelType.SONNET,
            "prompts": {}
        }, "chemi-con": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.FREQUENCY: primary_table_prompt
            }
        }, "elna": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.FREQUENCY: primary_table_prompt
            }
        }, "rubycon": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.FREQUENCY: primary_table_prompt
            }
        }, "panasonic": {
            "model": ModelType.SONNET,
            "prompts": {
                TableType.FREQUENCY: primary_table_prompt
            }
        }
    }

    def execute(self) -> str:
        """
        Clean and transform the table data.
        
        Returns:
            Cleaned and transformed CSV data
        """
        initial_content = [TextBlockParam(
            type="text",
            text=self.table_data,
            cache_control=CacheControlEphemeralParam(
                type="ephemeral"
            )
        )]
        
        # Get single response since we don't need continuation for cleaning
        response = next(self.iterate_thread(initial_content))
        return response.strip() 