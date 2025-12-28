import glob
import json
import os
import re

INPUT_DIR = "../../data/twhg_with_latlng_and_places/"
OUTPUT_DIR = "../../data/cleaned_twhg_with_latlng_and_places/"


def extract_features_and_description(description_text: str):
    """
    Extract a description segment before the structured features and the feature list itself.
    Returns: (description_before_features, extracted_feature_list)
    """
    extracted_feature_list = []

    pattern = re.compile(r"(?:^|\n)(?:Image\s*)?\d+\s+([^\n,]+).*?Tags:\s*([^\n]+)", re.IGNORECASE)
    first_match = None

    for match in pattern.finditer(description_text):
        if first_match is None:
            first_match = match

        room_raw = match.group(1).strip()
        tags_raw = match.group(2).strip()

        tag_list = []
        if tags_raw:
            for t in re.split(r'[、,]+', tags_raw):
                tag_cleaned = re.sub(r'\(.*?\)', '', t).strip()
                if tag_cleaned:
                    tag_list.append(tag_cleaned)
        if room_raw and tag_list:
            extracted_feature_list.append({
                "room": room_raw,
                "tag_list": tag_list
            })

    if first_match:
        description_before_features = description_text[:first_match.start()].strip()
    else:
        description_before_features = description_text.strip()

    return description_before_features, extracted_feature_list


def calculate_public_area_ratio(gross_area, interior_area):
    """
    Calculate the public area ratio (公設比) from gross and interior areas.
    Returns a rounded float or None when calculation is not possible.
    """

    def to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            normalized = value.replace(',', '').strip()
            if not normalized:
                return None
            return float(normalized)
        return None

    gross = to_float(gross_area)
    interior = to_float(interior_area)

    if gross is None or interior is None or gross <= 0:
        return None

    public_area = gross - interior
    if public_area < 0:
        return None

    return round(public_area / gross, 4)


def process_single_file(input_path, output_path):
    """Read a single file, clean it, and write to the output path"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        listing = data.get('listing', {})
        if not listing:
            print(f"[Warning] No listing data in {input_path}")
            return False

        # Get the original description
        raw_description = listing.get('description', '')

        description_before_features, extracted_feature_list = extract_features_and_description(raw_description)

        gross_area = listing.get('gross_area')
        interior_area = listing.get('interior_area')

        public_area_ratio = calculate_public_area_ratio(gross_area, interior_area)

        processed_item = {
            "original_url": data.get('url'),
            "property_id": listing.get('property_id'),
            "title": listing.get('title'),
            "total_price": listing.get('total_price'),
            "city": listing.get('city'),
            "district": listing.get('district'),
            "street": listing.get('street'),
            "property_type": listing.get('property_type'),
            "property_age": listing.get('property_age'),
            "gross_area": gross_area,
            "interior_area": interior_area,
            "public_area_ratio": public_area_ratio,
            "num_bedroom": listing.get('num_bedroom'),
            "num_bathroom": listing.get('num_bathroom'),
            "num_living_room": listing.get('num_living_room'),
            "transportation": listing.get('transportation'),
            "orientation": listing.get('orientation'),
            "picture_list": listing.get('picture_list'),
            "floor": listing.get('floor'),
            "total_floors": listing.get('total_floors'),
            "land_ownership_area": listing.get('land_ownership_area'),
            "property_usage": listing.get('property_usage'),
            "has_elevator": listing.get('has_elevator'),
            "parking_type": listing.get('parking_type'),
            "raw_description": raw_description,
            "description": description_before_features,
            "extracted_feature_list": extracted_feature_list
        }

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(processed_item, f_out, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"[Error] Failed to process {input_path}: {e}")
        return False


def main():
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} files in {INPUT_DIR}")

    processed_count = 0
    skipped_count = 0

    # 3. Process each file
    for input_path in json_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Skip Logic: Check if output already exists
        if os.path.exists(path=output_path):
            skipped_count += 1
            continue

        # Process each file
        success = process_single_file(input_path=input_path, output_path=output_path)
        if success:
            processed_count += 1

    print("-" * 30)
    print(f"Job Finished.")
    print(f"Total Files Found: {len(json_files)}")
    print(f"Processed New:     {processed_count}")
    print(f"Skipped Existing:  {skipped_count}")
    print(f"Output Directory:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
