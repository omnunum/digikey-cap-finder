import os
import re
import requests
import pandas as pd
import math
from dataclasses import dataclass, fields
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
from typing import Optional, Dict, List, Literal, Union, get_type_hints, DefaultDict, Any, TypeAlias
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator, ValidationError
import csv

from helpers import (
    get_parameter_value,
    get_variation_unit_price,
    parse_ripple_hf,
    parse_lifetime_temp,
    parse_diameter,
    parse_height,
    round_up_to_sigfig,
    build_search_payload,
    get_supported_voltages_for_cap,
    get_n_next_voltages_for_cap,
    create_cached_request_func,
    parse_voltage,
    parse_esr,
    parse_impedance
)


@dataclass
class Config:
    client_id: str
    client_secret: str
    token_url: str
    search_url: str
    temperature_rating: str
    min_quantity: int
    limit: int
    
    lifetime_temp_threshold: float
    weight_ripple: float
    weight_price: float
    weight_lifetime: float
    weight_diameter_penalty: float
    weight_voltage: float
    weight_esr: float  # New weight for ESR optimization
    
    allow_merge_with_higher_voltage: bool
    opportunistic_voltage_search: bool
    
    cache_dir: str
    input_source: str  # CSV file path or Google Sheets URL
    series_tables_file: str  # Path to the series tables processed file
    csv_output: str
    html_output: str

class SeriesData(BaseModel):
    series: str = Field(alias='Series')
    manufacturer: str = Field(alias='Manufacturer')
    capacitance: float = Field(alias='Capacitance')
    voltage: float = Field(alias='Voltage')
    esr: Optional[float] = Field(default=float('inf'), alias='ESR/Z 20°C@100kHz')
    ripple_100hz: Optional[float] = Field(default=0.0, alias='Ripple Current @120Hz')
    ripple_1khz: Optional[float] = Field(default=0.0, alias='Ripple Current @1kHz')
    ripple_10khz: Optional[float] = Field(default=0.0, alias='Ripple Current @10kHz')
    ripple_100khz: Optional[float] = Field(default=0.0, alias='Ripple Current @100kHz')
    diameter: Optional[float] = Field(default=0.0, alias='Case Size Diameter')
    length: Optional[float] = Field(default=0.0, alias='Case Size Length')
    
    class Config:
        # Allow population by alias
        allow_population_by_field_name = True
    
    @validator('series')
    def normalize_series(cls, v):
        return v.strip().lower() if v else v
# Type alias for our nested dictionary structure: series -> capacitance -> voltage -> SeriesData
SeriesDataMap: TypeAlias = DefaultDict[str, DefaultDict[float, Dict[float, SeriesData]]]

def read_series_data(series_tables_file: str) -> SeriesDataMap:
    """
    Read capacitor series data from the specified file using Pydantic for validation.
    Returns a nested defaultdict mapping series -> capacitance -> voltage -> SeriesData objects.
    This structure makes it easier to find substitutes with same capacitance but higher voltage.
    """
    # Create a nested defaultdict: series -> capacitance -> {voltage: SeriesData}
    series_data_map: SeriesDataMap = defaultdict(lambda: defaultdict(dict))
    skipped_rows = 0
    
    try:
        with open(series_tables_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for i, row in enumerate(reader):
                try:
                    # Convert empty strings to None
                    row_data = {k: (None if v == '' else v) for k, v in row.items()}
                    
                    # Let Pydantic handle the validation, type conversion and aliases
                    series_data = SeriesData.parse_obj(row_data)
                    # Store the data in the nested structure
                    series_data_map[series_data.series][series_data.capacitance][series_data.voltage] = series_data
                    
                except ValidationError as e:
                    skipped_rows += 1
                    print(f"Skipping row {i + 2}: {e}")
        
        print(f"Successfully read series data from {series_tables_file}")
    except Exception as e:
        print(f"Warning: Could not read series data: {e}")
        return defaultdict(lambda: defaultdict(dict))
    
    # Count the total number of entries
    num_entries = sum(sum(len(volt_dict) for volt_dict in cap_dict.values()) 
                      for cap_dict in series_data_map.values())
    print(f"Loaded {num_entries} series data entries. Skipped {skipped_rows} rows with missing required data.")
    return series_data_map

def read_input_data(input_source: str) -> pd.DataFrame:
    """
    Read data from either a CSV file or a Google Sheet URL.
    """
    try:
        if input_source.startswith(('http://', 'https://')) and 'docs.google.com/spreadsheets' in input_source:
            sheet_id = urlparse(input_source).path.split('/')[-2]
            sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df = pd.read_csv(sheet_url)
            print(f"Successfully read data from Google Sheet")
        else:
            df = pd.read_csv(input_source)
            print(f"Successfully read data from {input_source}")
    except Exception as e:
        raise ValueError(f"Error reading input data: {e}")
    
    column_mapping = {
        'Label': 'label',
        'Reference': 'label',
        'Capacitance µF': 'capacitance',
        'Voltage V': 'voltage',
        'Size mm': 'diameter',
        'Series': 'series',
        'Optimize Replacement For': 'optimize_for'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    if 'tight_fit' not in df.columns:
        df['tight_fit'] = True
    
    if 'optimize_for' not in df.columns:
        df['optimize_for'] = 'Ripple'  # Default to Ripple optimization
    else:
        df['optimize_for'] = df['optimize_for'].apply(
            lambda x: 'Impedance' if str(x).upper() == 'IMPEDANCE' else 'Ripple'
        )
    
    df['capacitance'] = pd.to_numeric(df['capacitance'], errors='coerce')
    df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')
    df['diameter'] = pd.to_numeric(df['diameter'], errors='coerce')
    
    df = df.dropna(subset=['capacitance', 'voltage', 'diameter'])
    
    if df.empty:
        raise ValueError("No valid data in input source after filtering")
    
    return df

def find_best_series_match(
    series: str, 
    capacitance: float, 
    voltage: float, 
    series_data_map: SeriesDataMap
) -> Optional[SeriesData]:
    """
    Find a matching series data entry for a given product.
    First tries an exact match, then looks for the same capacitance with higher voltage.
    
    Args:
        series: The product series name (lowercase)
        capacitance: The product capacitance in μF
        voltage: The product voltage rating
        series_data_map: The nested map of series data
        
    Returns:
        The matching SeriesData object or None if no match found
    """
    if not series or not series_data_map:
        return None
        
    # Get the series dictionary - returns empty defaultdict if series doesn't exist
    series_caps = series_data_map[series]
        
    # Only try exact capacitance match
    if not capacitance in series_caps:
        return None

    cap_voltages = series_caps[capacitance]    
    # Try exact voltage match
    if voltage in cap_voltages:
        return cap_voltages[voltage]
        
    # Look for next higher voltage option
    higher_voltages = [v for v in cap_voltages.keys() if v > voltage and v <= voltage * 1.2]
    if higher_voltages:
        # Get the next higher voltage
        return cap_voltages[min(higher_voltages)]
    
    return None


def compute_composite_scores(
    products, 
    config, 
    source_diameter, 
    quantity=1, 
    tight_fit=True, 
    optimize_for: Literal['Impedance', 'Ripple'] = 'Ripple',
    series_data_map: Optional[SeriesDataMap] = None
):
    """
    Compute composite scores for candidate products based on optimization preference.
    
    Scoring is based on:
    - High frequency (100kHz) ripple current rating
    - ESR/Z value
    - Lifetime
    - Price
    - Voltage rating
    - Diameter constraints
    
    For ripple optimization, high ripple current ratings are prioritized.
    For impedance optimization, low ESR/Z values are prioritized.
    """
    from math import log10, sqrt
    
    quantity_factor = min(1 + log10(max(1, quantity)), 5)
    
    raw_items = []
    for prod in products:
        variation_price = get_variation_unit_price(prod)
        if variation_price is None:
            continue
        
        # Parse ripple current, focusing on value at high frequency
        hf_text = get_parameter_value(prod, "Ripple Current @ High Frequency")
        ripple_rating, ripple_freq = parse_ripple_hf(hf_text)  # ripple_rating in mA, ripple_freq in kHz
        
        # For scoring purposes, we'll prefer products with ratings at/near 100kHz
        # but we'll store the actual values separately for the output
        raw_ripple = ripple_rating  # Using just mA value directly
        
        lifetime_text = get_parameter_value(prod, "Lifetime @ Temp.")
        lifetime, temp = parse_lifetime_temp(lifetime_text)
        raw_lifetime = sqrt(1 + lifetime) * max(0, (temp - config.lifetime_temp_threshold))
        
        raw_price = variation_price * quantity
        
        diameter_text = get_parameter_value(prod, "Size / Dimension")
        product_diameter = parse_diameter(diameter_text)
        
        voltage_text = get_parameter_value(prod, "Voltage - Rated")
        raw_voltage = parse_voltage(voltage_text) if voltage_text else 0.0
        
        series_text = prod.get("Series", {}).get("Name", "")
        series = series_text.lower() if series_text else ""
        
        # Get ESR, impedance and series data - we'll use whichever is available
        esr_text = get_parameter_value(prod, "ESR (Equivalent Series Resistance)")
        esr_rating = parse_esr(esr_text)
        esr_freq = 100  # Default to 100kHz for ESR measurements
        
        impedance_text = get_parameter_value(prod, "Impedance")
        impedance_rating = parse_impedance(impedance_text)
        impedance_freq = 100  # Default to 100kHz for impedance measurements
        
        # Find direct match in the series data map
        best_match = None
        series_esr = None
        
        if series_data_map and series:
            capacitance = 0.0
            cap_match = re.search(r'\d+\.?\d*', get_parameter_value(prod, "Capacitance"))
            if cap_match and (str_val := cap_match.group(0)):
                capacitance = float(str_val)
                
                # Use the simplified function to find a direct match
                best_match = find_best_series_match(
                    series, capacitance, raw_voltage, series_data_map
                )
                if best_match and best_match.esr is not None and best_match.esr < float('inf'):
                    series_esr = best_match.esr
        
        # Simple OR logic for resistance - use whatever is available
        # For display, we'll store all of them separately
        scoring_resistance = float('inf')
        if esr_rating is not None and esr_rating > 0:
            scoring_resistance = esr_rating
        elif impedance_rating is not None and impedance_rating > 0:
            scoring_resistance = impedance_rating / 1000.0  # Convert mOhm to Ohm for scoring
        elif series_esr is not None:
            scoring_resistance = series_esr
            esr_rating = series_esr  # Use series ESR for display too
        
        # For ripple current, use best_match data if available and better
        if best_match and best_match.ripple_100khz and (ripple_rating < 100 or ripple_rating < best_match.ripple_100khz):
            ripple_rating = best_match.ripple_100khz
            ripple_freq = 100  # 100kHz
            raw_ripple = ripple_rating  # Update raw_ripple too
        
        raw_items.append({
            "prod": prod,
            "composite": 0.0,
            "raw_ripple": raw_ripple,  # For scoring
            "ripple_rating": ripple_rating,  # Actual value in mA for output
            "ripple_freq": ripple_freq,  # Frequency in kHz for output
            "esr_rating": esr_rating,  # ESR value in ohms
            "esr_freq": esr_freq,  # ESR measurement frequency in kHz (normally 100kHz)
            "impedance_rating": impedance_rating,  # Impedance value in mOhms
            "impedance_freq": impedance_freq,  # Impedance measurement frequency in kHz
            "scoring_resistance": scoring_resistance,  # The resistance value used for scoring
            "raw_lifetime": raw_lifetime,
            "raw_price": raw_price,
            "raw_voltage": raw_voltage,
            "lifetime": lifetime,
            "temp": temp,
            "product_diameter": product_diameter,
            "unit_price": variation_price,
            "series": series,
            "series_data": best_match
        })
    
    if not raw_items:
        return []
    
    # Calculate maximums for normalization
    max_ripple = max(i["raw_ripple"] for i in raw_items) or 1
    
    # Collect valid resistance values (excluding infinite values)
    resistance_values = [i["scoring_resistance"] for i in raw_items if i["scoring_resistance"] < float('inf')]
    min_resistance = min(resistance_values) if resistance_values else 1
    
    max_lifetime = max(i["raw_lifetime"] for i in raw_items) or 1
    max_price = max(i["raw_price"] for i in raw_items) or 1
    max_voltage = max(i["raw_voltage"] for i in raw_items) or 1
    
    # Set scoring factors based on optimization preference
    ripple_factor = 1.0 if optimize_for == 'Ripple' else 0.5
    impedance_factor = 1.0 if optimize_for == 'Impedance' else 0.5
    
    scored = []
    for i in raw_items:
        # Normalize values for scoring
        nr = i["raw_ripple"] / max_ripple  # Higher ripple is better
        
        # Resistance (ESR or Impedance) is better when lower, so invert the score
        if resistance_values and i["scoring_resistance"] < float('inf'):
            ne = min_resistance / i["scoring_resistance"]
        else:
            ne = 0  # No resistance contribution if no data available
            
        nl = i["raw_lifetime"] / max_lifetime
        np_ = i["raw_price"] / max_price
        nv = i["raw_voltage"] / max_voltage
        
        # Compute composite score
        composite = (
            (config.weight_ripple * ripple_factor) * nr
            + (config.weight_esr * impedance_factor) * ne  # Use ESR weight for impedance too
            + config.weight_lifetime * nl
            - config.weight_price * np_ * quantity_factor
            + config.weight_voltage * nv
        )
        
        # Apply diameter penalty if needed
        threshold = source_diameter if tight_fit else source_diameter * 1.25
        if i["product_diameter"] is not None and i["product_diameter"] > threshold:
            diff = i["product_diameter"] - threshold
            composite -= diff * config.weight_diameter_penalty
        
        i["composite"] = composite
        scored.append(i)
    
    # Sort by composite score (highest first)
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


def search_capacitor(
    capacitance, 
    voltage, 
    source_diameter, 
    cached_request, 
    config: Config, 
    quantity, 
    tight_fit, 
    optimize_for: Literal['Impedance', 'Ripple'] = 'Ripple',
    series_data_map: Optional[SeriesDataMap] = None
):
    payload = build_search_payload(capacitance, config, voltage=voltage)
    try:
        data = cached_request(payload)
    except requests.HTTPError as e:
        print(f"Error requesting data for {payload['Keywords']}: {e}")
        return None
    
    products = data.get("Products", [])
    if not products:
        print(f"No products found for {payload['Keywords']}")
        return None

    scored = compute_composite_scores(
        products, 
        config, 
        source_diameter, 
        quantity, 
        tight_fit, 
        optimize_for,
        series_data_map
    )
    if not scored:
        print(f"No eligible products for {payload['Keywords']}")
        return None
    

    def create_product_dict(prod_data):
        product = prod_data["prod"]
        available = product.get("QuantityAvailable")
        try:
            available = int(available)
        except (ValueError, TypeError):
            return
        if available < quantity:
            return

        # ensure product has at least one single MOQ variation
        part_num = product.get("ManufacturerProductNumber", "N/A")
        for variation in product["ProductVariations"]:
            if variation.get("MinimumOrderQuantity") == 1:
                part_num = variation.get("DigiKeyProductNumber", part_num)
                break
        else:
            return
        
        diam_text = get_parameter_value(product, "Size / Dimension")
        ht_text = get_parameter_value(product, "Height - Seated (Max)")
        diam = parse_diameter(diam_text)
        height = parse_height(ht_text)
        
        product_url = product.get("ProductUrl", "")        
        customer_ref = ""
        if product_url:
            parts = product_url.rstrip("/").split("/")
            if parts:
                customer_ref = parts[-1]
        part_link = f'<a href="{product_url}" target="_blank">{part_num}</a>' if product_url else part_num
        
        # Format ripple rating with frequency
        ripple_rating = int(round(prod_data["ripple_rating"]))
        ripple_freq = int(round(prod_data["ripple_freq"]))
        ripple_display = f"{ripple_rating}mA\n@{ripple_freq}kHz" if ripple_rating > 0 and ripple_freq > 0 else ""
        
        # Get ESR and impedance values
        esr_rating = prod_data.get("esr_rating") 
        esr_freq = int(round(prod_data.get("esr_freq", 100)))
        impedance_rating = prod_data.get("impedance_rating", 0)
        impedance_freq = int(round(prod_data.get("impedance_freq", 100)))
        
        # Build ESR/Z display string
        esr_parts = []
        if esr_rating is not None and esr_rating > 0:
            esr_parts.append(f"ESR: {esr_rating}Ω")
        if impedance_rating and impedance_rating > 0:
            esr_parts.append(f"Z: {impedance_rating}mΩ")
        
        if esr_parts:
            # Add frequency to the display if we have any resistance values
            esr_display = "\n".join(esr_parts)
            # Use ESR frequency if available, otherwise default to impedance frequency
            freq = esr_freq if (esr_rating is not None and esr_rating > 0) else impedance_freq
            esr_display += f"\n@{freq}kHz"
        else:
            esr_display = ""
        
        return {
            "Capacitance": round_up_to_sigfig(capacitance, 1),
            "Voltage": round_up_to_sigfig(voltage, 1),
            "Manufacturer": product.get("Manufacturer", {}).get("Name", "N/A"),
            "Part Number Link": part_link,
            "PartNumber": part_num,
            "CustomerReference": customer_ref,
            "Ripple": ripple_display,
            "ESR": esr_display,
            "Lifetime": int(round(prod_data["lifetime"])),
            "Temp": int(round(prod_data["temp"])),
            "Diameter": round(diam, 2) if diam else "",
            "Height": round(height, 2) if height else "",
            "Price": round(prod_data["unit_price"], 2),
            "Composite": round(prod_data["composite"], 2),
            "OptimizeFor": optimize_for,
            "Series": prod_data.get("series", "").upper()
        }
    
    formatted_products = [
        pdict
        for p in scored 
        if (pdict := create_product_dict(p)) is not None
    ]

    return formatted_products

def merge_with_higher_voltage(specs_data):
    from collections import defaultdict
    final = []
    groups = defaultdict(list)
    for item in specs_data:
        cap_level = item["best"]["Capacitance"]
        groups[cap_level].append(item)
    
    for cap_level, items in groups.items():
        items.sort(key=lambda i: i["best"]["Voltage"])
        n = len(items)
        in_use = [True] * n
        
        for i_idx in range(n - 1):
            if not in_use[i_idx]:
                continue
            i_volt = items[i_idx]["best"]["Voltage"]
            i_qty  = items[i_idx]["quantity"]
            i_diam = items[i_idx]["best"]["Diameter"] or ""
            i_h    = items[i_idx]["best"]["Height"] or ""
            try:
                i_diam_f = float(i_diam)
                i_h_f    = float(i_h)
            except ValueError:
                continue
            for j_idx in range(i_idx + 1, n):
                if not in_use[j_idx]:
                    continue
                j_volt = items[j_idx]["best"]["Voltage"]
                if j_volt <= i_volt:
                    continue
                j_diam = items[j_idx]["best"]["Diameter"] or ""
                j_h    = items[j_idx]["best"]["Height"] or ""
                try:
                    j_diam_f = float(j_diam)
                    j_h_f    = float(j_h)
                except ValueError:
                    continue
                if abs(i_diam_f - j_diam_f) > 2:
                    continue
                if abs(i_h_f - j_h_f) > 2:
                    continue
                items[j_idx]["quantity"] += i_qty
                in_use[i_idx] = False
                break
        for idx in range(n):
            if in_use[idx]:
                final.append(items[idx])
    return final

def deduplicate_specs_data(specs_data):
    deduped = {}
    for item in specs_data:
        part = item["best"]["PartNumber"]
        label = item["best"].get("CustomerReference", "")

        if part not in deduped:
            deduped[part] = item.copy()
            continue

        deduped[part]["quantity"] += item["quantity"]
        # Combine labels from both items
        existing_label = deduped[part]["best"].get("CustomerReference", "")
        combined_labels = set(existing_label.split(";") if existing_label else [])
        combined_labels.update(label.split(";") if label else [])
        combined_labels = sorted(filter(None, combined_labels))
        deduped[part]["best"]["CustomerReference"] = ";".join(combined_labels)

        if item["best"]["Composite"] <= deduped[part]["best"]["Composite"]:
            continue

        # Combine labels when replacing best
        new_label = item["best"].get("CustomerReference", "")
        combined_labels += (new_label.split(";") if new_label else [])
        combined_labels = sorted(filter(None, combined_labels))
        deduped[part]["best"] = item["best"]
        deduped[part]["best"]["CustomerReference"] = ";".join(combined_labels)
    return list(deduped.values())

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config = Config(
        client_id=os.getenv("DIGIKEY_CLIENT_ID"),
        client_secret=os.getenv("DIGIKEY_CLIENT_SECRET"),
        token_url='https://api.digikey.com/v1/oauth2/token',
        search_url='https://api.digikey.com/products/v4/search/keyword',
        temperature_rating="85°C",
        min_quantity=5,
        limit=50,
        lifetime_temp_threshold=85.0,
        weight_ripple=3.0,
        weight_price=1.0,
        weight_lifetime=1.0,
        weight_diameter_penalty=2.0,
        weight_voltage=2.0,
        weight_esr=3.0,  # Add weight for ESR optimization
        allow_merge_with_higher_voltage=False,
        opportunistic_voltage_search=True,
        cache_dir=os.path.join(script_dir, "cache"),
        input_source="https://docs.google.com/spreadsheets/d/17cPtcG3J5juaralesjxz_KZUVDKJV5dGURIf7dJdRtU/edit?usp=sharing",
        series_tables_file=os.path.join(script_dir, "all_series_priority_data.csv"),
        csv_output=os.path.join(script_dir, "digikey_best_caps_final.csv"),
        html_output=os.path.join(script_dir, "digikey_best_caps_final.html")
    )
    
    # Read series tables data for enhanced scoring
    series_data_map = read_series_data(config.series_tables_file)
    
    cap_df = read_input_data(config.input_source)

    # Continue with the existing processing
    grouped = cap_df.groupby(["capacitance", "voltage", "diameter", "tight_fit", "optimize_for"]).agg(
        Labels=('label', lambda x: ";".join(sorted(set(x)))),
        Quantity=('label', 'size')
    ).reset_index()
    
    # Get access token (for all searches)
    resp = requests.post(
        config.token_url,
        data={
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "grant_type": "client_credentials"
        }
    )
    resp.raise_for_status()
    token_data = resp.json()
    if "access_token" not in token_data:
        print("Failed to obtain token:", token_data)
        return
    access_token = token_data["access_token"]
    
    # Create a cached request function bound to config and access token.
    cached_request = create_cached_request_func(config, access_token)
    
    # Build a dictionary of supported voltages keyed by capacitance.
    unique_caps = sorted(set(grouped["capacitance"]))
    cap_to_voltages = {}
    for cap in unique_caps:
        voltages = get_supported_voltages_for_cap(cap, cached_request, config)
        cap_to_voltages[cap] = voltages
        print(f"Capacitance {cap}µF supports voltages: {voltages}")
    
    specs_data = []
    for _, row in grouped.iterrows():
        cap = row["capacitance"]
        volt = row["voltage"]
        src_dia = float(row["diameter"])
        qty = row["Quantity"]
        tight_fit = str(row["tight_fit"]).strip().upper() == "TRUE"
        labels = row["Labels"]
        optimize_for = row["optimize_for"]
        
        if config.opportunistic_voltage_search:
            supported = cap_to_voltages.get(cap, [])
            next_voltages = get_n_next_voltages_for_cap(cap, volt, 2, supported)
            candidate_voltages = [volt] + next_voltages
            all_candidates = []
            for cand in candidate_voltages:
                out = search_capacitor(
                    cap, cand, src_dia, cached_request, config, 
                    qty, tight_fit, optimize_for, series_data_map
                )
                if out is not None:
                    all_candidates.extend(out)
            if not all_candidates:
                while (out := search_capacitor(
                    cap, volt, src_dia, cached_request, config, 
                    qty, tight_fit, optimize_for, series_data_map
                )) is None:
                    print(f"Trying to search for next nearest voltage for {cap}µF, {volt}V")
                    next_vs = get_n_next_voltages_for_cap(cap, volt, 1, cap_to_voltages.get(cap, []))
                    if not next_vs:
                        break
                    volt = next_vs[0]
                all_candidates.extend(out)
            all_candidates.sort(key=lambda x: x["Composite"], reverse=True)
            best_item = all_candidates[0]
            runner_ups = all_candidates[1:11]
        else:
            while (out := search_capacitor(
                cap, volt, src_dia, cached_request, config, 
                qty, tight_fit, optimize_for, series_data_map
            )) is None:
                print(f"Trying to search for next nearest voltage for {cap}µF, {volt}V")
                next_vs = get_n_next_voltages_for_cap(cap, volt, 1, cap_to_voltages.get(cap, []))
                if not next_vs:
                    break
                volt = next_vs[0]
            best_item = out[0]
            runner_ups = out[1:11]
        
        best_item["Quantity"] = qty
        best_item["CustomerReference"] = labels
        best_item["OptimizeFor"] = optimize_for
        
        specs_data.append({
            "best": best_item,
            "runner_ups": runner_ups,
            "quantity": qty
        })
    
    if config.allow_merge_with_higher_voltage:
        final_data = merge_with_higher_voltage(specs_data)
    else:
        final_data = specs_data
    
    final_data = deduplicate_specs_data(final_data)
    
    final_csv_rows = []
    for item in final_data:
        best = item["best"]
        best_dict = dict(best)
        best_dict["Quantity"] = item["quantity"]
        final_csv_rows.append(best_dict)
    
    df_csv = pd.DataFrame(final_csv_rows)
    df_csv.to_csv(config.csv_output, index=False)
    print(f"Saved CSV to {config.csv_output}")
    
    try:
        df_csv["Price"] = pd.to_numeric(df_csv["Price"], errors="coerce")
        df_csv["Quantity"] = pd.to_numeric(df_csv["Quantity"], errors="coerce")
    except Exception as e:
        pass
    
    total_cost = (df_csv["Price"] * df_csv["Quantity"]).sum()
    unique_caps = df_csv.shape[0]
    total_caps = df_csv["Quantity"].sum()
    total_cost_str = f"${total_cost:,.2f}"
    
    env = Environment(loader=FileSystemLoader(os.path.join(script_dir, "templates")), autoescape=False)
    template = env.get_template("final_report.html.j2")
    
    html_content = template.render(
        specs_data=final_data,
        total_cost_str=total_cost_str,
        unique_caps=unique_caps,
        total_caps=total_caps
    )
    out_html = config.html_output
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML saved to {out_html}")

if __name__ == "__main__":
    main()
