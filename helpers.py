import os
import math
import json
import requests
import hashlib

voltage_list = [
    5, 3, 4, 6, 7, 8, 10, 12, 15, 16, 20, 25, 28, 30, 35, 40, 42, 45, 50, 55, 
    56, 60, 63, 65, 70, 71, 75, 80, 90, 95, 100, 110, 120, 125, 150, 160, 175, 
    180, 200, 210, 220, 225, 230, 250, 280, 300, 315, 330, 350, 360, 375, 385, 
    400, 415, 420, 440, 450, 475, 500, 525, 550, 560, 570, 575, 580, 600, 630, 
    650, 700, 750
]

def get_n_next_voltages(voltage, n):
    return get_n_next_values(voltage_list, voltage, n)

def get_n_next_values(values, starting_value, n):
    """
    Get the next n entries from the list after a matching starting value.
    """
    try:
        # Find the index of the starting value
        index = values.index(starting_value)
        
        # Slice the list to get the next n entries
        next_values = values[index + 1: index + 1 + n]
        
        return next_values
    except ValueError:
        # If the input value is not found, return an empty list
        return []

def round_up_to_sigfig(number: float | int, rounded_to: int) -> float | int:
    return round(number, rounded_to) if number != int(number) else int(number)

# These parsing helpers handle the product's parameter extraction & numeric conversions.
def get_parameter_value(product, parameter_name):
    """Returns the product's parameter ValueText for a given parameter_name (case-insensitive)."""
    for param in product.get("Parameters", []):
        if param.get("ParameterText", "").lower() == parameter_name.lower():
            return param.get("ValueText", None)
    return None

def parse_ripple_hf(ripple_text):
    """
    Parse "Ripple Current @ High Frequency" string like '60 mA @ 100 kHz' or '3.99 A @ 100 kHz'.
    Returns (mA, freq_in_kHz). If parsing fails, returns (0, 0).
    """
    if not ripple_text:
        return (0, 0)
    try:
        parts = ripple_text.split("@")
        current_str = parts[0].strip()
        if "mA" in current_str:
            mA = float(current_str.split("mA")[0].strip())
        elif "A" in current_str:
            mA = float(current_str.split("A")[0].strip()) * 1000
        else:
            mA = 0
        freq_str = parts[1].strip()
        freq = float(freq_str.split("kHz")[0].strip())
        return (mA, freq)
    except:
        return (0, 0)

def parse_impedance(imp_text):
    """
    Parses e.g.:
      - "1.37 Ohms"  -> 1370.0 mOhms
      - "900 mOhms"  -> 900.0  mOhms
    Defaults to 0.0 if parsing fails.
    """
    if not imp_text:
        return 0.0
    try:
        # Split on whitespace; first token is the numeric part, e.g. "1.37" or "900"
        parts = imp_text.split()
        numeric_str = parts[0].strip()
        value = float(numeric_str)
        
        # Convert everything to lower case for unit detection
        text_lower = imp_text.lower()
        
        if "mohms" in text_lower:
            # e.g. "900 mOhms" -> 900.0
            return value
        else:
            # e.g. "1.37 ohms" -> multiply by 1000 to get mOhms
            # If "mOhms" is NOT found, assume it's in ohms
            return value * 1000.0
    except:
        return 0.0

def parse_lifetime_temp(lifetime_text):
    """Parses e.g. '2000 Hrs @ 105°C' -> (2000, 105). Returns (0, 0) if fails."""
    if not lifetime_text:
        return (0, 0)
    try:
        parts = lifetime_text.split("@")
        lifetime = float(parts[0].split("Hrs")[0].strip())
        temp = float(parts[1].replace("°C", "").strip())
        return (lifetime, temp)
    except:
        return (0, 0)

def parse_esr(esr_text):
    """Returns ESR as float, or 0 on failure."""
    if not esr_text:
        return 0
    try:
        return float(esr_text)
    except:
        return 0

def parse_diameter(size_text):
    """Extract the diameter (in mm) from e.g. '0.630" Dia (16.00mm)' -> 16.0. Returns None if not found."""
    if not size_text:
        return None
    try:
        start = size_text.find("(")
        end = size_text.find("mm", start)
        if start == -1 or end == -1:
            return None
        val = size_text[start+1:end].strip()
        return float(val)
    except:
        return None

def parse_height(height_text):
    """Extract the height (in mm) from e.g. '1.063" (27.00mm)' -> 27.0. Returns None if not found."""
    if not height_text:
        return None
    try:
        start = height_text.find("(")
        end = height_text.find("mm", start)
        if start == -1 or end == -1:
            return None
        val = height_text[start+1:end].strip()
        return float(val)
    except:
        return None


def get_variation_unit_price(product):
    """
    Iterate over product['ProductVariations']. For each variation that has MinimumOrderQuantity <= 1,
    check its StandardPricing for an entry with BreakQuantity == 1.
    Return the min price found, or None if none found.
    """
    variations = product.get("ProductVariations", [])
    prices = []
    for var in variations:
        moq = var.get("MinimumOrderQuantity")
        if moq is not None and moq <= 1:
            standard_pricing = var.get("StandardPricing", [])
            for sp in standard_pricing:
                if sp.get("BreakQuantity") == 1:
                    try:
                        prices.append(float(sp["UnitPrice"]))
                    except:
                        pass
    if prices:
        return min(prices)
    return None


def make_cached_request(url, payload, headers, cache_dir):
    """
    Creates an MD5 hash of (url + sorted JSON payload). If a file with that hash
    is found in cache_dir, we load and return it. Otherwise, make the request,
    store the result, and return it.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    cache_key_str = url + json.dumps(payload, sort_keys=True)
    cache_hash = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()
    cache_file_path = os.path.join(cache_dir, cache_hash + ".json")
    
    if os.path.isfile(cache_file_path):
        print(f"Using cached response for hash {cache_hash}")
        with open(cache_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"No cache found for hash {cache_hash}, requesting from API...")
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    
    with open(cache_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data
