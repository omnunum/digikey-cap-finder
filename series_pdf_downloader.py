import io
import os
import re
import unicodedata

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber
import PyPDF2
import pytesseract
import pandas as pd
import requests

from bs4 import BeautifulSoup

from helpers import make_cached_request

@dataclass
class Series:
    name: str
    raw_pages: list[pdfplumber.page.Page]
    raw_tables: Optional[list[list[list[str | None]]]] = None
    processed_frame: Optional[pd.DataFrame] = None
    dissipation_frame: Optional[pd.DataFrame] = None

@dataclass
class Catalog:
    name: str
    url: str
    new_page_triggers: list[str]
    starting_page: Optional[int] = None
    ending_page: Optional[int] = None
    pdf_data: Optional[io.BytesIO] = None

def fetch_nichicon_data():
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

def normalize_text(text):
    """
    Normalize text by converting full-width characters to ASCII and removing extra spaces.
    """
    # Convert full-width characters to ASCII
    normalized = unicodedata.normalize('NFKC', text)
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    return normalized

def extract_series_names(new_page_triggers, text):
    """
    Extract series names from text using trigger strings and regex patterns.
    Returns the series name that appears earliest in the text.
    For bipolar series, includes the full "(Bi-polar)" designation.
    
    Parameters:
    new_page_triggers (list[str]): List of strings that indicate where to look for series names
    text (str): Text to search for series names
    
    Returns:
    str: Extracted series name or None if not found
    """
    patterns = [
        r"([A-Z\-]{2,}[0-9]*[\s\n]+\(Bi-?polar\))\s+[sS]eries",
        r"([A-Z\-]{2,}[0-9]*)[\s\n]+[sS]eries",
        r"[sS]eries[\s\n]+([A-Z\-]{2,}[0-9]*)",
    ]

    trigger_index = -1
    
    for trigger in new_page_triggers:
        next_trigger_index = text.find(trigger, trigger_index + 1)
        if next_trigger_index == -1:
            return None  # Not all triggers found in order
        trigger_index = next_trigger_index
        
    # Find all matches for all patterns
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.group(1)))
    
    # Return the earliest match if any found
    if matches:
        # Sort by position and return the first match's series name
        return min(matches, key=lambda x: x[0])[1]
    
    return None

def find_series_in_catalog(catalog: Catalog):
    """
    Extracts series information from a PDF file within the specified page range.

    Parameters:
    catalog (Catalog): A Catalog object containing:
        - pdf_data: The PDF content as BytesIO
        - trigger_text: String used to identify pages to search for series
        - starting_page: Page number to start searching from
        - ending_page: Optional page number to end searching at

    Returns:
    list: A list of Series objects containing the series name and the raw pages associated with it.
    """
    if catalog.pdf_data is None:
        raise ValueError(f"ERROR: {catalog.name} has no PDF data")

    # Dictionary to store series data
    series_dict: dict[str, Series] = defaultdict(lambda: Series(name='', raw_pages=[]))
    current_series = None  # Track the current series we're collecting pages for

    
    # Open PDF with pdfplumber
    with pdfplumber.open(catalog.pdf_data) as pdf:
        # Determine the page range
        start_page = catalog.starting_page or 0
        end_page = catalog.ending_page or len(pdf.pages)
        
        # Start from specified page where series for our type of data begins
        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num]
            text = page.extract_text()

            if not text:
                # If we can't extract text via pdfplumber, use Tesseract OCR
                image = page.to_image(resolution=300)
                text = pytesseract.image_to_string(image.original)
            if not text:
                print(f"ERROR: Could not extract text from page {page_num} even after OCRing")
                continue

            normalized_text = normalize_text(text)
            normalized_trigger_text = [normalize_text(t) for t in catalog.new_page_triggers]
            
            # Try to extract series name from the page
            name = extract_series_names(normalized_trigger_text, normalized_text)
            
            
            # If we found a new series, start collecting its pages
            if name:
                normalized_name = normalize_text(name)
                current_series = normalized_name
                if not series_dict[normalized_name].name:  # Only set name if it hasn't been set yet
                    series_dict[normalized_name].name = name
                    print(f"Found series: {name}")
                series_dict[normalized_name].raw_pages.append(page)
                continue
            # If we didn't find a series name but we have a current series and the trigger text is present
            elif current_series:
                # Check if the current series name appears in the text
                if current_series in normalized_text:
                    print(f"Found continuation page for series {current_series} on page {page_num}")
                    series_dict[current_series].raw_pages.append(page)
                    continue
            print(f"Could not find series on page {page_num}")
            current_series = None  # Reset current series if we don't find trigger text
    
    print(f"\nFound {len(series_dict)} series in total.")
    for series in series_dict:
        print(f"{series}: {len(series_dict[series].raw_pages)} pages")
    
    return list(series_dict.values())


def main():
    """Main function to extract data from Panasonic catalog."""
    # Create output directory if it doesn't exist
    os.makedirs('series_pdfs', exist_ok=True)
    catalogs = [
        Catalog(
            name="Chemi-Con",
            url="https://www.chemi-con.co.jp/products/relatedfiles/capacitor/catalog/al-all-e.pdf",
            new_page_triggers=["MINIATURE ALUMINUM ELECTROLYTIC CAPACITORS"],
            starting_page=150,
            ending_page=239
        ),
        # Catalog(
        #     name="Panasonic",
        #     url="https://industrial.panasonic.com/cdbs/www-data/pdf/RDF0000/ast-ind-152839.pdf",
        #     new_page_triggers=["Aluminum Electrolytic Capacitors", "Radial Lead Type"],
        #     starting_page=14,
        # )
        # , Catalog(
        #     name="Rubycon",
        #     url="https://www.rubycon.co.jp/wp-content/uploads/catalog/aluminum-catalog.pdf",
        #     new_page_triggers=["ＲＡＤＩＡＬ ＬＥＡＤ ＡＬＵＭＩＮＵＭ ＥＬＥＣＴＲＯＬＹＴＩＣ ＣＡＰＡＣＩＴＯＲＳ"],
        #     starting_page=42,
        #     ending_page=112
        # ), Catalog(
        #     name="Elna",
        #     url="https://www.elna.co.jp/wp-content/uploads/2024/10/catalog_23-24_e.pdf",
        #     new_page_triggers=["Aluminum Electrolytic Capacitors"],
        #     starting_page=85,
        #     ending_page=122
        # )
    ]
    for c in catalogs:
        response = requests.get(c.url)
        response.raise_for_status()
        # Second pass to process each series
        with io.BytesIO(response.content) as pdf_content:
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            c.pdf_data = pdf_content
            for s in find_series_in_catalog(c):
                # Create a new PDF writer for this series
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add all pages for this series
                for page in s.raw_pages:
                    # pdfplumber page numbers are 0-based, PyPDF2 expects 0-based
                    pdf_writer.add_page(pdf_reader.pages[page.page_number - 1])
                
                # Save the series PDF
                output_path = Path(f"series_pdfs") / f"{c.name.lower()}" / f"{s.name}.pdf"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)

if __name__ == "__main__":
    main()