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
    parse_diameter,
    parse_height,
    round_up_to_sigfig,
    build_search_payload,
    get_supported_voltages_for_cap,
    get_n_next_voltages_for_cap,
    create_cached_request_func,
    parse_voltage
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
    weight_voltage: float          # new weight for voltage bonus
    
    allow_merge_with_higher_voltage: bool
    opportunistic_voltage_search: bool
    
    cache_dir: str
    csv_input: str
    csv_output: str
    html_output: str

def compute_composite_scores(products, config, source_diameter, quantity=1, tight_fit=True):
    """
    Compute composite scores for candidate products. In addition to ripple, lifetime,
    price, and a diameter penalty, a bonus is added for higher rated voltage.
    Only products with QuantityAvailable >= quantity and not marked as Marketplace are considered.
    """
    from math import log10, sqrt
    
    quantity_factor = min(1 + log10(max(1, quantity)), 5)
    
    raw_items = []
    for prod in products:
        if prod.get("MarketPlace", False):
            continue
        
        available = prod.get("QuantityAvailable")
        try:
            available = int(available)
        except (ValueError, TypeError):
            continue
        if available < quantity:
            continue
        
        variation_price = get_variation_unit_price(prod)
        if variation_price is None:
            continue
        
        hf_text = get_parameter_value(prod, "Ripple Current @ High Frequency")
        mA, freq = parse_ripple_hf(hf_text)
        raw_ripple = mA * freq
        
        lifetime_text = get_parameter_value(prod, "Lifetime @ Temp.")
        lifetime, temp = parse_lifetime_temp(lifetime_text)
        raw_lifetime = sqrt(1 + lifetime) * max(0, (temp - config.lifetime_temp_threshold))
        
        raw_price = variation_price * quantity
        
        diameter_text = get_parameter_value(prod, "Size / Dimension")
        product_diameter = parse_diameter(diameter_text)
        
        voltage_text = get_parameter_value(prod, "Voltage - Rated")
        raw_voltage = parse_voltage(voltage_text) if voltage_text else 0.0
        
        raw_items.append({
            "prod": prod,
            "composite": 0.0,
            "raw_ripple": raw_ripple,
            "raw_lifetime": raw_lifetime,
            "raw_price": raw_price,
            "raw_voltage": raw_voltage,
            "mA": mA,
            "freq": freq,
            "lifetime": lifetime,
            "temp": temp,
            "product_diameter": product_diameter,
            "unit_price": variation_price
        })
    
    if not raw_items:
        return []
    
    max_ripple = max(i["raw_ripple"] for i in raw_items) or 1
    max_lifetime = max(i["raw_lifetime"] for i in raw_items) or 1
    max_price = max(i["raw_price"] for i in raw_items) or 1
    max_voltage = max(i["raw_voltage"] for i in raw_items) or 1
    
    scored = []
    for i in raw_items:
        nr = i["raw_ripple"] / max_ripple
        nl = i["raw_lifetime"] / max_lifetime
        np_ = i["raw_price"] / max_price
        nv = i["raw_voltage"] / max_voltage
        composite = (
            config.weight_ripple * nr +
            config.weight_lifetime * nl -
            config.weight_price * np_ * quantity_factor +
            config.weight_voltage * nv
        )
        threshold = source_diameter if tight_fit else source_diameter * 1.25
        if i["product_diameter"] is not None and i["product_diameter"] > threshold:
            diff = i["product_diameter"] - threshold
            composite -= diff * config.weight_diameter_penalty
        
        i["composite"] = composite
        scored.append(i)
    
    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored

def search_capacitor(capacitance, voltage, source_diameter, cached_request, config: Config, quantity, tight_fit):
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
    
    scored = compute_composite_scores(products, config, source_diameter, quantity, tight_fit)
    if not scored:
        print(f"No eligible products for {payload['Keywords']}")
        return None
    
    best = scored[0]
    runner_ups = scored[1:]
    
    def create_product_dict(prod_data):
        diam_text = get_parameter_value(prod_data["prod"], "Size / Dimension")
        ht_text = get_parameter_value(prod_data["prod"], "Height - Seated (Max)")
        diam = parse_diameter(diam_text)
        height = parse_height(ht_text)
        
        product = prod_data["prod"]
        product_url = product.get("ProductUrl", "")
        part_num = product.get("ManufacturerProductNumber", "N/A")
        if "ProductVariations" in product:
            for variation in product["ProductVariations"]:
                if variation.get("MinimumOrderQuantity") == 1:
                    part_num = variation.get("DigiKeyProductNumber", part_num)
                    break
        customer_ref = ""
        if product_url:
            parts = product_url.rstrip("/").split("/")
            if parts:
                customer_ref = parts[-1]
        part_link = f'<a href="{product_url}" target="_blank">{part_num}</a>' if product_url else part_num
        
        return {
            "Capacitance": round_up_to_sigfig(capacitance, 1),
            "Voltage": round_up_to_sigfig(voltage, 1),
            "Manufacturer": product.get("Manufacturer", {}).get("Name", "N/A"),
            "Part Number Link": part_link,
            "PartNumber": part_num,
            "CustomerReference": customer_ref,
            "Ripple": int(round(prod_data["raw_ripple"])),
            "Lifetime": int(round(prod_data["lifetime"])),
            "Temp": int(round(prod_data["temp"])),
            "Diameter": round(diam, 2) if diam else "",
            "Height": round(height, 2) if height else "",
            "Price": round(prod_data["unit_price"], 2),
            "Composite": round(prod_data["composite"], 2)
        }
    
    best_dict = create_product_dict(best)
    runner_up_dicts = [create_product_dict(ru) for ru in runner_ups]
    
    return {"best": best_dict, "runner_ups": runner_up_dicts}

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
        weight_price: float
        weight_lifetime: float
        weight_diameter_penalty: float
        weight_voltage: float
        allow_merge_with_higher_voltage: bool
        opportunistic_voltage_search: bool
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
        weight_price=1.0,
        weight_lifetime=2.0,
        weight_diameter_penalty=2.0,
        weight_voltage=1.0,
        allow_merge_with_higher_voltage=False,
        opportunistic_voltage_search=True,
        cache_dir=os.path.join(script_dir, "digikey_cache"),
        csv_input=os.path.join(script_dir, "cap_list.csv"),
        csv_output=os.path.join(script_dir, "digikey_best_caps_final.csv"),
        html_output=os.path.join(script_dir, "digikey_best_caps_final.html")
    )
    
    # Read CSV (columns: label, capacitance, voltage, diameter, tight_fit)
    cap_df = pd.read_csv(config.csv_input)
    grouped = cap_df.groupby(["capacitance", "voltage", "diameter", "tight_fit"]).agg(
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
        
        if config.opportunistic_voltage_search:
            supported = cap_to_voltages.get(cap, [])
            next_voltages = get_n_next_voltages_for_cap(cap, volt, 2, supported)
            candidate_voltages = [volt] + next_voltages
            all_candidates = []
            for cand in candidate_voltages:
                out = search_capacitor(cap, cand, src_dia, cached_request, config, qty, tight_fit)
                if out is not None:
                    all_candidates.append(out["best"])
                    all_candidates.extend(out["runner_ups"])
            if not all_candidates:
                while (out := search_capacitor(cap, volt, src_dia, cached_request, config, qty, tight_fit)) is None:
                    print(f"Trying to search for next nearest voltage for {cap}µF, {volt}V")
                    next_vs = get_n_next_voltages_for_cap(cap, volt, 1, cap_to_voltages.get(cap, []))
                    if not next_vs:
                        break
                    volt = next_vs[0]
                all_candidates = [out["best"]] + out["runner_ups"]
            all_candidates.sort(key=lambda x: x["Composite"], reverse=True)
            best_item = all_candidates[0]
            runner_ups = all_candidates[1:11]
        else:
            while (out := search_capacitor(cap, volt, src_dia, cached_request, config, qty, tight_fit)) is None:
                print(f"Trying to search for next nearest voltage for {cap}µF, {volt}V")
                next_vs = get_n_next_voltages_for_cap(cap, volt, 1, cap_to_voltages.get(cap, []))
                if not next_vs:
                    break
                volt = next_vs[0]
            best_item = out["best"]
            runner_ups = out["runner_ups"][:10]
        
        best_item["Quantity"] = qty
        best_item["CustomerReference"] = labels
        
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
