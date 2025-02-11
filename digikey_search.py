import os
import requests
import pandas as pd
import math
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader

from helpers import (
    get_parameter_value,
    make_cached_request,
    get_variation_unit_price,
    parse_ripple_hf,
    parse_lifetime_temp,
    parse_impedance,
    parse_esr,
    parse_diameter,
    parse_height,
    round_up_to_sigfig,
    get_n_next_voltages
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
    weight_impedance: float
    weight_price: float
    weight_lifetime: float
    weight_diameter_penalty: float
    
    allow_merge_with_higher_voltage: bool  # <-- new boolean config
    
    cache_dir: str
    csv_input: str
    csv_output: str
    html_output: str

def compute_composite_scores(products, config, source_diameter, quantity=1):
    """
    Same as your existing function, but with diameter penalty logic.
    ...
    """
    from math import log, log10
    
    # For example: quantity_factor = min(1 + log10(max(1, quantity)), 5)
    quantity_factor = min(1 + log10(max(1, quantity)), 5)
    
    raw_items = []
    for prod in products:
        variation_price = get_variation_unit_price(prod)
        if variation_price is None:
            continue
        
        # parse fields
        hf_text = get_parameter_value(prod, "Ripple Current @ High Frequency")
        mA, freq = parse_ripple_hf(hf_text)
        raw_ripple = mA * freq
        
        lifetime_text = get_parameter_value(prod, "Lifetime @ Temp.")
        lifetime, temp = parse_lifetime_temp(lifetime_text)
        
        # Example: sqrt(1 + lifetime) * max(0, temp - threshold)
        raw_lifetime = math.sqrt(1 + lifetime) * max(0, (temp - config.lifetime_temp_threshold))
        
        imp = parse_impedance(get_parameter_value(prod, "Impedance"))
        esr = parse_esr(get_parameter_value(prod, "ESR (Equivalent Series Resistance)"))
        raw_impedance = imp + esr
        
        raw_price = variation_price * quantity
        
        diameter_text = get_parameter_value(prod, "Size / Dimension")
        product_diameter = parse_diameter(diameter_text)
        
        raw_items.append({
            "prod": prod,
            "composite": 0.0,  # we'll compute below
            "raw_ripple": raw_ripple,
            "raw_lifetime": raw_lifetime,
            "raw_impedance": raw_impedance,
            "raw_price": raw_price,
            "mA": mA,
            "freq": freq,
            "lifetime": lifetime,
            "temp": temp,
            "product_diameter": product_diameter,
            "unit_price": variation_price
        })
    
    if not raw_items:
        return []
    
    # normalization
    max_ripple = max(i["raw_ripple"] for i in raw_items) or 1
    max_lifetime = max(i["raw_lifetime"] for i in raw_items) or 1
    max_impedance = max(i["raw_impedance"] for i in raw_items) or 1
    max_price = max(i["raw_price"] for i in raw_items) or 1
    
    scored = []
    for i in raw_items:
        nr = i["raw_ripple"] / max_ripple
        nl = i["raw_lifetime"] / max_lifetime
        ni = i["raw_impedance"] / max_impedance
        np = i["raw_price"] / max_price
        
        composite = (
            config.weight_ripple * nr +
            config.weight_lifetime * nl -
            config.weight_impedance * ni -
            config.weight_price * np * quantity_factor
        )
        
        # diameter penalty
        if i["product_diameter"] is not None and i["product_diameter"] > source_diameter:
            diff = i["product_diameter"] - source_diameter
            composite -= diff * config.weight_diameter_penalty
        
        i["composite"] = composite
        scored.append(i)
    
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


def search_capacitor(capacitance, voltage, source_diameter, access_token, config: Config, quantity):
    """
    Returns best+runner_ups for a single CSV line.
    """    
    # Build the payload
    search_payload = {
        "Keywords": f"{round_up_to_sigfig(capacitance, 1)}µF {round_up_to_sigfig(voltage, 1)}V {config.temperature_rating}",
        "Limit": config.limit,
        "Offset": 0,
        "MinimumQuantityAvailable": config.min_quantity,
        "FilterOptionsRequest": {
            "ParameterFilterRequest": {
                "CategoryFilter": {"Id": "58"},
                "ParameterFilters": [
                    {
                        "ParameterId": 16,
                        "FilterValues": [{"Id": "392320"}]
                    }
                ]
            },
            "ManufacturerFilter": [
                {"Id": mid} 
                for mid in ["565","399","493","1189","732"]
            ]
        },
        "SortOptions": {
            "Field": "2260",
            "SortOrder": "Descending"
        }
    }
    headers = {
        "X-DIGIKEY-Client-Id": config.client_id,
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "application/json",
    }
    
    try:
        data = make_cached_request(config.search_url, search_payload, headers, config.cache_dir)
    except requests.HTTPError as e:
        print(f"Error requesting data for {search_payload['Keywords']}: {e}")
        return None
    
    products = data.get("Products", [])
    if not products:
        print(f"No products found for {search_payload['Keywords']}")
        return None
    
    scored = compute_composite_scores(products, config, source_diameter, quantity)
    if not scored:
        print(f"No eligible products for {search_payload['Keywords']}")
        return None
    
    best = scored[0]
    runner_ups = scored[1:]

    def create_product_dict(prod_data):
        # parse diam/height
        diam_text = get_parameter_value(prod_data["prod"], "Size / Dimension")
        ht_text   = get_parameter_value(prod_data["prod"], "Height - Seated (Max)")
        
        diam = parse_diameter(diam_text)
        height = parse_height(ht_text)
        
        product_url = prod_data["prod"].get("ProductUrl", "")
        part_num = prod_data["prod"].get("ManufacturerProductNumber", "N/A")
        part_link = f'<a href="{product_url}" target="_blank">{part_num}</a>' if product_url else part_num
        
        return {
            "Capacitance": round_up_to_sigfig(capacitance, 1),
            "Voltage": round_up_to_sigfig(voltage, 1),
            "Manufacturer": prod_data["prod"].get("Manufacturer", {}).get("Name", "N/A"),
            "Part Number Link": part_link,
            "Ripple": int(round(prod_data["raw_ripple"])),
            "Impedance": 0 if prod_data["raw_impedance"] <= 0.001 else round(prod_data["raw_impedance"],2),
            "Lifetime": int(round(prod_data["lifetime"])),
            "Temp": int(round(prod_data["temp"])),
            "Diameter": round(diam,2) if diam else "",
            "Height": round(height,2) if height else "",
            "Price": round(prod_data["unit_price"],2),
            "Composite": round(prod_data["composite"],2)
        }
    
    best_dict = create_product_dict(best)
    runner_up_dicts = [create_product_dict(ru) for ru in runner_ups]
    
    return {
        "best": best_dict,
        "runner_ups": runner_up_dicts
    }


def merge_with_higher_voltage(specs_data):
    """
    Within each group of items having the same Capacitance, merge lower-voltage items into
    higher-voltage items if both diameter and height are within 2mm.

    Steps:
      1) Group specs_data by best["Capacitance"].
      2) Within each group, sort by best["Voltage"] ascending.
      3) For each item i from low to high voltage, try to merge it into a higher-voltage item j
         if (j_volt > i_volt) and abs(i_diam - j_diam) <= 2 and abs(i_height - j_height) <= 2.
         If merged, add item i's quantity to item j and drop item i.

    Returns the final merged list of items.
    """
    from collections import defaultdict

    final = []
    groups = defaultdict(list)
    for item in specs_data:
        # item["best"] has "Capacitance", "Voltage", "Diameter", "Height", etc.
        cap_level = item["best"]["Capacitance"]
        groups[cap_level].append(item)
    
    for cap_level, items in groups.items():
        # sort by voltage ascending
        items.sort(key=lambda i: i["best"]["Voltage"])
        n = len(items)
        in_use = [True]*n
        
        # For each item i, see if it can merge into a higher-voltage item j
        for i_idx in range(n-1):
            if not in_use[i_idx]:
                continue

            i_volt = items[i_idx]["best"]["Voltage"]
            i_qty  = items[i_idx]["quantity"]
            i_diam = items[i_idx]["best"]["Diameter"] or ""
            i_h    = items[i_idx]["best"]["Height"] or ""
            
            # Convert to float if possible; skip if blank
            try:
                i_diam_f = float(i_diam)
                i_h_f    = float(i_h)
            except ValueError:
                continue  # no size info => skip merging

            # Look for j > i
            for j_idx in range(i_idx+1, n):
                if not in_use[j_idx]:
                    continue

                j_volt = items[j_idx]["best"]["Voltage"]
                # If j's voltage is not strictly greater, skip
                if j_volt <= i_volt:
                    continue

                j_diam = items[j_idx]["best"]["Diameter"] or ""
                j_h    = items[j_idx]["best"]["Height"] or ""
                
                try:
                    j_diam_f = float(j_diam)
                    j_h_f    = float(j_h)
                except ValueError:
                    continue  # skip if missing dimension

                # Must be within 2 mm for diameter & height
                if abs(i_diam_f - j_diam_f) > 2:
                    continue
                if abs(i_h_f - j_h_f) > 2:
                    continue

                # If we get here, we can merge i into j
                items[j_idx]["quantity"] += i_qty
                in_use[i_idx] = False
                break  # done merging item i
        
        # Add items that remain in use
        for idx in range(n):
            if in_use[idx]:
                final.append(items[idx])
    
    return final



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    @dataclass
    class LocalConfig:
        client_id: str
        client_secret: str
        token_url: str
        search_url: str
        temperature_rating: str
        min_quantity: int
        limit: int
        lifetime_temp_threshold: float
        weight_ripple: float
        weight_impedance: float
        weight_price: float
        weight_lifetime: float
        weight_diameter_penalty: float
        allow_merge_with_higher_voltage: bool
        cache_dir: str
        csv_input: str
        csv_output: str
        html_output: str
    
    config = LocalConfig(
        client_id=os.getenv("DIGIKEY_CLIENT_ID"),
        client_secret=os.getenv("DIGIKEY_CLIENT_SECRET"),
        token_url='https://api.digikey.com/v1/oauth2/token',
        search_url='https://api.digikey.com/products/v4/search/keyword',
        temperature_rating="105°C",
        min_quantity=5,
        limit=50,
        lifetime_temp_threshold=85.0,
        weight_ripple=3.0,
        weight_impedance=2.0,
        weight_price=1.0,
        weight_lifetime=2.0,
        weight_diameter_penalty=2.0,
        allow_merge_with_higher_voltage=False,  # <-- turn merging on
        cache_dir=os.path.join(script_dir, "digikey_cache"),
        csv_input=os.path.join(script_dir, "cap_list.csv"),
        csv_output=os.path.join(script_dir, "digikey_best_caps_final.csv"),
        html_output=os.path.join(script_dir, "digikey_best_caps_final.html")
    )
    
    cap_df = pd.read_csv(config.csv_input)
    # we assume columns: (capacitance, voltage, diameter)
    grouped = cap_df.groupby(["capacitance", "voltage", "diameter"]).size().reset_index(name="Quantity")
    
    # get token
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
    
    specs_data = []
    for _, row in grouped.iterrows():
        cap = row["capacitance"]
        volt = row["voltage"]
        src_dia = float(row["diameter"])
        qty = row["Quantity"]
        
        while (out := search_capacitor(cap, volt, src_dia, access_token, config, qty)) is None:
            print(f"Trying to search for next nearest voltage")
            volt = get_n_next_voltages(volt, 1)[0]
        
        best_item = out["best"]
        best_item["Quantity"] = qty
        runner_ups = out["runner_ups"]
        
        # store
        specs_data.append({
            "best": best_item,
            "runner_ups": runner_ups,
            "quantity": qty
        })
    
    # Possibly merge with higher voltage
    if config.allow_merge_with_higher_voltage:    
        final_data = merge_with_higher_voltage(specs_data)
    else:
        final_data = specs_data
    
    # Rebuild final data for CSV
    final_csv_rows = []
    for item in final_data:
        best = item["best"]
        # put quantity in best item
        best_dict = dict(best)
        best_dict["Quantity"] = item["quantity"]
        final_csv_rows.append(best_dict)
    
    df_csv = pd.DataFrame(final_csv_rows)
    df_csv.to_csv(config.csv_output, index=False)
    print(f"Saved CSV to {config.csv_output}")
    
    # Summaries
    try:
        df_csv["Price"]   = pd.to_numeric(df_csv["Price"], errors="coerce")
        df_csv["Quantity"] = pd.to_numeric(df_csv["Quantity"], errors="coerce")
    except:
        pass
    
    total_cost = (df_csv["Price"] * df_csv["Quantity"]).sum()
    unique_caps = df_csv.shape[0]
    total_caps  = df_csv["Quantity"].sum()
    total_cost_str = f"${total_cost:,.2f}"
    
    # Jinja2 for HTML
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
