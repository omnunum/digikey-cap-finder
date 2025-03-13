import os
import math
import json
import requests
import hashlib
import re

def round_up_to_sigfig(number: float | int, rounded_to: int) -> float | int:
    if math.isnan(number):
        return number
    return round(number, rounded_to) if number != int(number) else int(number)

def build_search_payload(capacitance, config, voltage=None):
    """
    Build a search payload for a given capacitance and optional voltage.
    
    If voltage is None, the Keywords string will include only the capacitance and temperature rating.
    Otherwise, it includes capacitance, voltage, and temperature rating.
    """
    keywords = (
        f"{str(round_up_to_sigfig(capacitance, 1 if capacitance > 1 else 2)).replace('.0', '')}µF" 
        + (f" {round_up_to_sigfig(voltage, 1)}V" if voltage else "")
    )
   
    payload = {
        "Keywords": keywords,
        "Limit": config.limit,
        "Offset": 0,
        "FilterOptionsRequest": {
            "CategoryFilter": [{"Id": "3"}],
            "ParameterFilterRequest": {
                "CategoryFilter": {"Id": "58"},
                "ParameterFilters": [
                    {"ParameterId": 69, "FilterValues": [{"Id": "411897"}]},
                    {"ParameterId": 16, "FilterValues": [{"Id": "392320"}]}
                ]
            },
            "ManufacturerFilter": [{"Id": mid} for mid in ["10", "565", "399", "493", "1189"]],
            "MinimumQuantityAvailable": config.min_quantity,
        },
        "SortOptions": {
            "Field": "2260",
            "SortOrder": "Descending"
        }
    }
    return payload

def make_cached_request(url, payload, headers, cache_dir, response_type='json', params=None):
    """
    Creates an MD5 hash of (url + sorted JSON payload + sorted query params). If a file with that hash
    is found in cache_dir, we load and return it. Otherwise, make the request,
    store the result, and return it.
    
    Args:
        url: The URL to make the request to
        payload: The request payload/data
        headers: Request headers
        cache_dir: Directory to store cached responses
        response_type: Type of response to expect/cache ('json' or 'html')
        params: Optional query string parameters
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # Include params in cache key if provided
    cache_key_str = url + json.dumps(payload, sort_keys=True)
    if params:
        cache_key_str += json.dumps(params, sort_keys=True)
        
    cache_hash = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()
    extension = '.json' if response_type == 'json' else '.html'
    cache_file_path = os.path.join(cache_dir, cache_hash + extension)
    
    if os.path.isfile(cache_file_path):
        print(f"Using cached response for hash {cache_hash}")
        with open(cache_file_path, "r", encoding="utf-8") as f:
            if response_type == 'json':
                return json.load(f)
            return f.read()
    
    print(f"No cache found for hash {cache_hash}, requesting from API...")
    resp = requests.post(url, headers=headers, json=payload, params=params)
    resp.raise_for_status()
    
    with open(cache_file_path, "w", encoding="utf-8") as f:
        if response_type == 'json':
            data = resp.json()
            json.dump(data, f, indent=2)
            return data
        else:
            f.write(resp.text)
            return resp.text

def create_cached_request_func(config, access_token):
    """
    Returns a function that, when given a payload, will call make_cached_request
    using the provided config and access token (bound into the headers).
    """
    headers = {
         "X-DIGIKEY-Client-Id": config.client_id,
         "authorization": f"Bearer {access_token}",
         "content-type": "application/json",
         "accept": "application/json",
    }
    def cached_request(payload):
         return make_cached_request(config.search_url, payload, headers, config.cache_dir)
    return cached_request

def parse_voltage(voltage_text):
    """
    Parse a voltage string like "350 V" and return the numeric value (e.g. 350.0).
    Returns None if parsing fails.
    """
    if not voltage_text:
        return None
    try:
        match = re.search(r"([\d\.]+)", voltage_text)
        if match:
            return float(match.group(1))
    except Exception as e:
        return None

def get_supported_voltages_for_cap(capacitance, cached_request, config):
    """
    For a given capacitance (in µF), perform a search (without specifying voltage)
    to retrieve the supported voltage ratings from FilterOptions.ParametricFilters
    where ParameterId == 2079.
    
    Returns a sorted list of numeric voltage values.
    """
    payload = build_search_payload(capacitance, config, voltage=None)
    response = cached_request(payload)
    
    supported_voltages = []
    filter_options = response.get("FilterOptions", {})
    parametric_filters = filter_options.get("ParametricFilters", [])
    for pf in parametric_filters:
        if not pf.get("ParameterId") == 2079:
            continue
        for fv in pf.get("FilterValues", []):
            value_text = fv.get("ValueName", "")
            match = re.search(r"(\d+)", value_text)
            if match:
                supported_voltages.append(int(match.group(1)))
    return sorted(set(supported_voltages))

def get_n_next_voltages_for_cap(capacitance, current_voltage, n, supported_voltages):
    """
    Given a sorted list of supported voltages (numeric) for a particular capacitance,
    return the next n voltages that are greater than current_voltage.
    """
    next_voltages = []
    for v in supported_voltages:
        if v > current_voltage:
            next_voltages.append(v)
        if len(next_voltages) >= n:
            break
    return next_voltages

# Parsing helpers

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
        parts = imp_text.split()
        numeric_str = parts[0].strip()
        value = float(numeric_str)
        text_lower = imp_text.lower()
        if "mohms" in text_lower:
            return value
        else:
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
    """
    Parse the ESR value from a string like "2.5 Ohm" or "100 mOhm" and return a float in Ohms.
    Returns None if no data is available or parsing fails.
    
    Common formats:
      - "0.1 Ohm" -> 0.1 Ohms
      - "100 mOhm" -> 0.1 Ohms (converts mOhm to Ohm)
    """
    if not esr_text:
        return None
    
    try:
        # Extract numeric value using regex
        match = re.search(r"([\d\.]+)", esr_text)
        if match:
            value = float(match.group(1))
            # Check if it's in mOhm
            if "mohm" in esr_text.lower():
                value *= 0.001  # Convert from mOhm to Ohm
            return value
    except Exception:
        return None

def parse_diameter(size_text):
    """Extract the diameter (in mm) from e.g. '0.630\" Dia (16.00mm)' -> 16.0. Returns None if not found."""
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
    """Extract the height (in mm) from e.g. '1.063\" (27.00mm)' -> 27.0. Returns None if not found."""
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
