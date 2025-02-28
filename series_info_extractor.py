"""
PDF table extraction using Claude 3.5 Sonnet with direct PDF handling.
"""
import os
import io
import base64
import re
from typing import List, Dict, Union, TypedDict
import pandas as pd
import pdfplumber
from pathlib import Path
from enum import Enum
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam, DocumentBlockParam, Base64PDFSourceParam, CacheControlEphemeralParam


class ModelType(Enum):
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-7-sonnet-20250219"


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
    
class ClaudeExtractor:
    def __init__(self, api_key: str, model: ModelType):
        """Initialize the extractor with Claude API key."""
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def extract_tables(self, pdf_block_param: Union[TextBlockParam, DocumentBlockParam], system_prompt: str, initial_message: str) -> Dict[str, pd.DataFrame]:
        """Extract tables from PDF using Claude, with precise token counting."""
        messages: List[MessageParam] = []
        complete_response_text = []
        end_of_document = False
        
        input_message = MessageParam(
            role="user",
            content=[
                pdf_block_param, 
                TextBlockParam(
                    type="text",
                    text=initial_message
                )
            ]
        )
        messages.append(input_message)

        while not end_of_document:
            try:
                response = self.client.messages.create(
                    model=self.model.value,
                    max_tokens=8192,
                    temperature=0,
                    system=[
                        TextBlockParam(
                            type="text",
                            text=system_prompt
                        )
                    ],
                    messages=messages
                )
                
                # Get response text
                response_text = ""
                for block in response.content:
                    if block.type == "text":
                        response_text = block.text
                        break
                
                # Add to conversation history and count cumulative tokens
                messages.append(MessageParam(
                    role="assistant",
                    content=[TextBlockParam(
                        type="text",
                        text=response_text
                    )]
                ))

                print(
                    f"Input size: {response.usage.input_tokens} tokens\n",
                    f"Response size: {response.usage.output_tokens} tokens\n"
                )

                # Alert if response is smaller than expected
                if response.usage.output_tokens < 4000:  # You can adjust this threshold
                    print(f"Warning: Response is using less than half of available capacity")
                    
                    # Optional: Log details about small responses for analysis
                    print("Response preview:")
                    print(response_text[:200] + "...")
                
                if "<end_of_document>" in response_text:
                    end_of_document = True
                    response_text = response_text.replace("<end_of_document>", "").strip()
                
                complete_response_text.append(response_text)
                if end_of_document:
                    continue

                messages.append(MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(
                            type="text",
                            text="<continue>"
                        )
                    ]
                ))
                
            except Exception as e:
                print(f"Error in API call: {e}")
                break

        full_text = "\n".join(complete_response_text)
        tables = parse_xml_tables(full_text)
        return tables

    def save_results(self, data: Dict[str, pd.DataFrame], output_path: Path) -> None:
        """Save extracted data to JSON and CSV files."""
        for name, table in data.items():
            # Convert to DataFrame and save CSV
            try:
                df = pd.DataFrame(table)
                csv_path = output_path.with_name(f"{output_path.stem}_{name}.csv")
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Error converting {name} to CSV: {e}")

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
"""
        },
    }
    for manufacturor, config in configuration.items():
        # Initialize extractor
        extractor = ClaudeExtractor(api_key, model=config["model"])
        # Process each PDF in series_pdfs directory
        pdf_dir = Path(f"series_pdfs/{manufacturor}")
        for pdf_path in pdf_dir.glob("*.pdf"):
            try:
                # Check if tables already exist for this series
                output_dir = base_output_dir / manufacturor
                output_path = output_dir / pdf_path.stem
                if output_dir.exists() and any(output_dir.glob(f"{pdf_path.stem}*.csv")):
                    print(f"\nSkipping {pdf_path.name} - tables already exist in {output_dir}")
                    continue

                print(f"\nProcessing {pdf_path.name}")
                
                # Count input tokens for monitoring total costs
                if config["model"] == ModelType.SONNET:
                    pdf_block = DocumentBlockParam(
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
                # Load PDF as base64
                print(f"Loaded PDF: {pdf_path.name}")

                # Extract tables
                data = extractor.extract_tables(pdf_block, system_prompt, config["prompt"])
                print(f"Extracted {len(data)} table entries")

                # Save results
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / pdf_path.stem
                extractor.save_results(data, output_path)
                print(f"Results saved to {output_path}")

            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue

if __name__ == "__main__":
    main()
