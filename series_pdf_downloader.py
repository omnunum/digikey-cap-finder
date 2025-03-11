import io
import os
import re
import unicodedata

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable

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
    specific_patterns: Optional[list[str]] = None
    starting_page: Optional[int] = None
    ending_page: Optional[int] = None
    pdf_data: Optional[io.BytesIO] = None


def normalize_text(text):
    """
    Normalize text by converting full-width characters to ASCII and removing extra spaces.
    """
    # Convert full-width characters to ASCII
    normalized = unicodedata.normalize('NFKC', text)
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    return normalized

def extract_series_names(new_page_triggers: list[str], text: str, specific_patterns: Optional[list[str]] = None) -> Optional[str]:
    """
    Extract series names from text using trigger strings and regex patterns.
    Returns the series name that appears earliest in the text.
    For bipolar series, includes the full "(Bi-polar)" designation.
    
    Parameters:
    new_page_triggers (list[str]): List of strings that indicate where to look for series names
    text (str): Text to search for series names
    specific_patterns (Optional[list[str]]): Additional catalog-specific regex patterns
    
    Returns:
    Optional[str]: Extracted series name or None if not found
    """
    # Standard patterns for finding series names
    standard_patterns = [
        r"([A-Z\-]{2,}[0-9]*[\s\n]+\(Bi-?polar\))\s+[sS]eries",
        r"([A-Z\-]{2,}[0-9]*)[\s\n]+[sS]eries",
        r"[sS]eries[\s\n]+([A-Z\-]{2,}[0-9]*)",
        *(specific_patterns or [])
    ]

    trigger_index = -1
    
    for trigger in new_page_triggers:
        next_trigger_index = text.find(trigger, trigger_index + 1)
        if next_trigger_index == -1:
            return None  # Not all triggers found in order
        trigger_index = next_trigger_index
        
    # Find all matches for all patterns
    matches = []
    for pattern in standard_patterns:
        for match in re.finditer(pattern, text):
            # Ensure we have a capture group
            if match.lastindex and match.lastindex >= 1:
                series_name = match.group(1).replace(" ", "")
                matches.append((match.start(), series_name))
    
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
            text = page.dedupe_chars().extract_text()

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
            name = extract_series_names(normalized_trigger_text, normalized_text, catalog.specific_patterns)
            
            
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

def download_pdf_links(url: str, output_dir: Path, transform_basename: Optional[Callable[[str], str]] = None) -> List[str]:
    """
    Extract all links from a PDF and download any PDFs found in those links.
    
    Parameters:
        url: URL of the source PDF to extract links from
        output_dir: Directory to save downloaded PDFs to
        transform_basename: Optional function to transform the basename of downloaded files
        
    Returns:
        List of PDF links found
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the source PDF
    response = requests.get(url)
    response.raise_for_status()
    
    # Extract base URL for handling relative paths
    base_url = url.rsplit('/', 1)[0] + '/'
    
    pdf_links_set: set[str] = set()
    
    # Extract hyperlinks
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        for page in pdf.pages:
            if not page.hyperlinks:
                continue
            for link in page.hyperlinks:
                uri = link.get('uri')
                if not (uri and isinstance(uri, str) and uri.lower().endswith('.pdf')):
                    continue
                # Handle relative URLs by combining with base URL
                if not uri.startswith(('http://', 'https://')):
                    full_url = base_url + uri
                else:
                    full_url = uri
                pdf_links_set.add(full_url)
    
    # Convert to list for downloading
    pdf_links = list(pdf_links_set)
    print(f"Found {len(pdf_links)} unique PDF links")
    
    # Download each PDF link
    for link in pdf_links:
        try:
            # Get filename from URL
            filename = link.split('/')[-1]
            
            # Apply basename transformation if provided
            if transform_basename:
                base_name, extension = os.path.splitext(filename)
                new_base_name = transform_basename(base_name)
                output_filename = new_base_name + extension
            else:
                output_filename = filename
                
            pdf_response = requests.get(link)
            pdf_response.raise_for_status()
            
            # Save to the output directory
            output_path = output_dir / output_filename
            with open(output_path, 'wb') as f:
                f.write(pdf_response.content)
            print(f"Downloaded: {link} as {output_filename}")
        except Exception as e:
            print(f"Failed to download {link}: {e}")
    
    return pdf_links

def main():
    """Main function to extract data from Panasonic catalog."""
    # Create output directory if it doesn't exist
    output_dir = Path("series_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download PDFs from links in the Nichicon catalog
    nichicon_catalog_url = "https://www.nichicon.co.jp/english/products/pdf/e-alm_mini_list_c.pdf"
    print(f"Downloading PDFs from links in {nichicon_catalog_url}")
    
    # Nichicon has an older catalog that has links to different PDFs
    #   instead of having all the series in one file
    pdf_links = download_pdf_links(
        nichicon_catalog_url,
        output_dir / "nichicon",
        lambda basename: basename.replace('e-', '').upper()
    )
    print(f"Found {len(pdf_links)} PDF links")
    
    catalogs = [
        Catalog(
            name="Nichicon",
            url="https://www.nichicon.co.jp/english/_assets/images/products/catalog/corporate/digital/e-lead.pdf",
            new_page_triggers=["ALUMINUM ELECTROLYTIC CAPACITORS"],
            specific_patterns=[
                # Nichicon-specific pattern for series like "UBX" at the end of a numbered list
                r"1 2 3 4 5 6 7 8 9 10 11 (([A-Z\-]\s?){3})\s+",
                r"ALUMINUM ELECTROLYTIC CAPACITORS (([A-Z\-]\s?){3})\s+"
            ]
        ),
        # Catalog(
        #     name="Chemi-Con",
        #     url="https://www.chemi-con.co.jp/products/relatedfiles/capacitor/catalog/al-all-e.pdf",
        #     new_page_triggers=["MINIATURE ALUMINUM ELECTROLYTIC CAPACITORS"],
        #     starting_page=150,
        #     ending_page=239
        # ),
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
                output_path = output_dir / f"{c.name.lower()}" / f"{s.name}.pdf"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)

if __name__ == "__main__":
    main()