"""
PDF table extraction using Claude 3.5 Sonnet with direct PDF handling.
"""
import os
import io
import base64
import re
from typing import List, Dict, Union
import pandas as pd
import pdfplumber
from pathlib import Path
from anthropic.types import TextBlockParam, DocumentBlockParam, Base64PDFSourceParam, CacheControlEphemeralParam

from anthropic_api import AnthropicAPI, ModelType


def load_pdf_text(pdf_path: Path) -> str:
    """Extract text content from PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text

def load_pdf_data(pdf_path: Path) -> str:
    """Load PDF and encode as base64."""
    with pdf_path.open("rb") as pdf_file:
        binary_data = pdf_file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        return base64_encoded_data.decode("utf-8")

def parse_xml_tables(text: str) -> Dict[str, pd.DataFrame]:
    """Parse tables with XML tags and csv format."""
    tables = {}
    
    # Find all table sections
    sections = re.finditer(r'<title>(.*?)</title>\s*<data>(.*?)</data>', text, re.DOTALL)
    
    for section in sections:
        title = section.group(1).strip()
        table_text = section.group(2).strip()
        
        # parse csv table
        df = pd.read_csv(io.StringIO(table_text))
        if title not in tables:
            tables[title] = df
        else:
            tables[title] = pd.concat([tables[title], df], ignore_index=True)
    
    return tables
    
class TableExtractor:
    def __init__(self, api: AnthropicAPI):
        """Initialize the extractor with an AnthropicAPI instance."""
        self.api = api

    def extract_tables(
        self, 
        pdf_path: Path,
        system_prompt: str, 
        initial_message: str,
        model: ModelType,
        max_tokens: int = 8192
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract tables from PDF using Claude with document extraction logic.
        
        Uses the message_iterator from AnthropicAPI to handle the conversation flow.
        The iterator yields responses, and this method decides whether to continue
        based on whether "<end_of_document>" marker is detected.
        
        Args:
            pdf_path: Path to the PDF file
            system_prompt: System prompt to guide Claude's behavior
            initial_message: Initial message requesting table extraction
            model: The Claude model to use for this conversation
            max_tokens: Maximum tokens for Claude response
        """
        # Prepare PDF content based on model
        pdf_block: DocumentBlockParam | TextBlockParam
        if model == ModelType.SONNET:
            pdf_block =  DocumentBlockParam(
                type="document",
                source=Base64PDFSourceParam(
                    type="base64",
                    media_type="application/pdf",
                    data=load_pdf_data(pdf_path)
                ),
                cache_control=CacheControlEphemeralParam(
                    type="ephemeral"
                )
            )
        else:
            pdf_block = TextBlockParam(
                type="text",
                text=load_pdf_text(pdf_path),
                cache_control=CacheControlEphemeralParam(
                    type="ephemeral"
                )
            )
        
        initial_content = [
            pdf_block, 
            TextBlockParam(
                type="text",
                text=initial_message
            )
        ]
        
        # Create message iterator and collect responses
        complete_response_text = []
        message_gen = self.api.message_iterator(
            system_prompt=system_prompt,
            initial_content=initial_content,
            model=model,
            max_tokens=max_tokens
        )
        
        # Get first response and iterate through responses
        response_text = next(message_gen)
        
        # Continue until end of document or error
        while True:
            # Check for end of document marker
            if "<end_of_document>" in response_text:
                # End of document reached, clean response and finish
                response_text = response_text.replace("<end_of_document>", "").strip()
                complete_response_text.append(response_text)
                break
            
            # Save this response
            complete_response_text.append(response_text)
            
            # Request next chunk with "continue" message
            response_text = message_gen.send("<continue>")
        
        # Parse collected responses
        full_text = "\n".join(complete_response_text)
        tables = parse_xml_tables(full_text)
        
        if not tables:
            print("No tables found in extracted content")
            
        return tables

def main():
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Create output directory
    base_output_dir = Path("series_tables")

    # System prompt for table extraction
    system_prompt = """
Directive:
    - You are an AI model specifically designed to extract tables from PDFs efficiently and completely by responding with tagged data.

Critical Requirements:
    - Never repeat values you have already extracted. Always extract new values.
    - While adhering to the previous uniqueness rule, maximize your response size by extracting as much as possible from the document in each response.
    - When a table spans multiple pages, extract ALL rows before moving to the next table
    - Output <end_of_document> when you have reached the end of all tables. Do not use any other way to communicate you have extracted all data.
    - Do not output any text outside of <title> and <data> tags.  I do not want any other communication outside of tags.
    - Do not hallucinate tables that don't exist, read until the end of the document and then just stop.
    - Never summarize or aggregate, extract exact values only

Response Formatting Details:
    - Each table must be tagged: <title>{name}</title> and <data>{data}</data>
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
    - Be careful to make sure that the amount of values and column names match, there should not be any fully empty columns.  
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
    configuration = {
        "nichicon": {
            "model": ModelType.SONNET,
            "prompt": """
Please extract the following tables:
- Required: "Frequency coefficient of rated ripple current" 
    - Frequency
    - Capacitance (µF) 
    - Coefficient
    - Voltage (V)
Notes about the table:
    - If the table looks like a graph/chart instead of a table, ignore it for now and do not include it in the output
    - The capacitance value looks like the left-most "column" of the table but is acually a vertical header
    - The Frequency is the headers above each column
        - Remove the "or more" from the frequency values
        - Keep the units of the frequency values (Hz/kHz)
    - Capacitance is almost always in the table but is occasionally missing
        - Keep the "{value} to {value}" format for the capacitance values if it is present
        - If it is missing, do not include it in the output
    - If voltage is present and exits across multiple rows, broadcast the values to all relevantrows
        - Keep the "{value} to {value}" format for the voltage values if it is present
"""
        },
        "chemi-con": {
            "model": ModelType.SONNET,
            "prompt": """
Please extract the following tables:
- Optional: "Dissipation Factor (tan δ)" (on the first page if it exists)
- Required: "STANDARD RATINGS" (on subsequent pages), possible columns include (in no particular order and not exhaustive)
    - WV (Vdc) 
    - Cap (µF) 
    - Case size φD×L(mm) 
    - tanδ 
    - Rated ripple current (mArms/ {temp}, {frequency}) 
    - Part No.
    - ESR (Ω max./{frequency}) {temp}
Notes about "STANDARD RATINGS" table:
    - For some of the ratings, there are notes about the temp and frequency earlier in the document.  Please take those notes and integrate them into the relevant column names.
        - For instance, the temp and frequency of the tan value is noted earlier in the document in the dissipation factor table.
"""
        }, 
        "samsung": {
            "model": ModelType.SONNET,
            "prompt": """
Samsung uses a format for their standard ratings table that is different from the other manufacturers, where the data is broken
up into multiple tables.  Please extract all of the tables and combine them into a single table as per the specs below.
Please extract the only the following tables:
- Optional: "Dissipation Factor (tan δ)" (on the first page if it exists)
    - Required columns (always present):
        - tanδ {temp} {frequency}
        - Rated Voltage (Vdc) 
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
Notes about "STANDARD RATINGS" table:
    - We want the input tables joined together to form a single table, so make sure to join the values in the tables by voltage value (not code) and capacitance value (not code)
    - Most rows will have all of the columns filled, but some rows may be missing values.
    - For some of the ratings, there are notes about the temp and frequency earlier in the document.  Please take those notes and integrate them into the relevant column names.
        - For instance, the temp and frequency of the tan value is noted earlier in the document in the dissipation factor table.
    - Pay close attention to which rating values belong to which voltage and capacitance values.
        - Some ratings will not exist at a given voltage and capacitance, and that is okay, ignore them.
        - Take extra time to get the first value of each voltage correct so that it has the correct capacitance.
"""
        },
        "kemet": {
            "model": ModelType.SONNET,
            "prompt": """
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
        }, 
        "elna": {
            "model": ModelType.SONNET,
            "prompt": """
Please extract the following tables:
    - Optional: "Tangent of loss angle (tan δ)"
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

Do not extract any other tables.

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
    """},
        "panasonic": {
            "model": ModelType.SONNET,
            "prompt": """
Please extract the following tables:
- Optional: "Dissipation Factor (MAX)" 
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
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
"""
        },
        "rubycon": {
            "model": ModelType.SONNET,
            "prompt": """
Please extract the following tables:
- Optional: "Dissipation Factor (MAX)" 
    - Required columns (always present): 
        - Rated Voltage
        - tan δ {temp} {frequency}
- Required: "STANDARD SIZE" 
    - Required columns (always present): 
        - Rated voltage (V)
        - Rated capacitance (μF)
        - Size øD x L
    - Optional columns (non exhaustive):
        - Rated ripple current (mArms) {temp} {frequency}
        - ESR {temp} {frequency} 
        - Impedance {temp} {frequency}
Tables will not always have these columns, but if they are present, please extract them.
For all formats:
    - Around the Standard Ratings table, there are notes about the temp and frequency of some of the ratings.  
        - Please take those notes and integrate them into the relevant column names.
    - Additionally, rating context can be found in the SPECIFICATIONS table (like with the tan value temp and frequency)
"""
        },
    }
    
    for manufacturer, config in configuration.items():
        # Initialize API client
        api_client = AnthropicAPI(api_key)
        extractor = TableExtractor(api_client)
        
        # Process each PDF in series_pdfs directory
        pdf_dir = Path(f"series_pdfs/{manufacturer}")
        for pdf_path in pdf_dir.glob("*.pdf"):
            try:
                # Check if tables already exist for this series
                output_dir = base_output_dir / manufacturer
                output_path = output_dir / pdf_path.stem
                if output_dir.exists() and any(output_dir.glob(f"{pdf_path.stem}*.csv")):
                    print(f"\nSkipping {pdf_path.name} - tables already exist in {output_dir}")
                    continue

                print(f"\nProcessing {pdf_path.name}")
                print(f"Loaded PDF: {pdf_path.name}")

                # Extract tables - now passing the model as well
                data = extractor.extract_tables(
                    pdf_path, 
                    system_prompt, 
                    config["prompt"],
                    config["model"]
                )
                print(f"Extracted {len(data)} table entries")

                # Save results
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / pdf_path.stem
                api_client.save_dataframes(data, output_path)
                print(f"Results saved to {output_path}")

            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue

if __name__ == "__main__":
    main()
