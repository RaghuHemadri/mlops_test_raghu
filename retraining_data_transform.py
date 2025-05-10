import os
import pandas as pd
import json
import shutil
from datetime import datetime

RAW_DIR = "/data_raw"
TRANSFORMED_DIR = "/data_transformed"
ARCHIVE_DIR = "/data_archive"
VERSION_FILE = os.path.join(TRANSFORMED_DIR, "version_tracker.txt")

# Step 1: Determine next version number
if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        last_version = int(f.read().strip())
else:
    existing_versions = [
        int(d.replace("v", "")) for d in os.listdir(TRANSFORMED_DIR)
        if os.path.isdir(os.path.join(TRANSFORMED_DIR, d)) and d.startswith("v") and d[1:].isdigit()
    ]
    last_version = max(existing_versions, default=0)

next_version = last_version + 1
version_tag = f"v{next_version}"

# Update tracker
with open(VERSION_FILE, "w") as f:
    f.write(str(next_version))

# Create output and archive version folders
version_dir = os.path.join(TRANSFORMED_DIR, version_tag)
versioned_archive_dir = os.path.join(ARCHIVE_DIR, version_tag)
os.makedirs(version_dir, exist_ok=True)
os.makedirs(versioned_archive_dir, exist_ok=True)

# Step 2: Read & process raw data
records = []
archived_files = []

for fname in os.listdir(RAW_DIR):
    if fname.endswith(".json"):
        file_path = os.path.join(RAW_DIR, fname)
        try:
            with open(file_path, "r") as f:
                records.extend(json.load(f))
                archived_files.append(fname)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON file {fname}: {e}")

        # Move file to archive after reading
        shutil.move(file_path, os.path.join(versioned_archive_dir, fname))

if not records:
    print("No new retraining data to process.")
    exit(0)

# Step 3: Clean the data
df = pd.DataFrame(records)
df = df.dropna(subset=["question", "answer"])
df = df[~df["question"].str.strip().eq("")]
df = df[~df["answer"].str.strip().eq("")]
df = df.drop(columns=["symptoms", "timestamp"], errors="ignore")

# Step 4: Save transformed data
data_out_path = os.path.join(version_dir, "retraining_data.json")
df.to_json(data_out_path, orient="records", lines=True)
print(f"Saved cleaned data to: {data_out_path}")

# Step 5: Write metadata
metadata = {
    "version": version_tag,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "record_count": len(df),
    "archived_files": archived_files
}
metadata_path = os.path.join(version_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote metadata to: {metadata_path}")
