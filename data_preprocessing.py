import os
import pandas as pd
from datasets import load_from_disk
import json 

# Load MedQuAD from saved Arrow format
ds = load_from_disk("/data/data/raw-dataset")
df = ds["train"].to_pandas()

# Keep only required columns
cols = ["synonyms", "question_type", "question", "question_focus", "answer"]
df = df[cols]

# Missing/blank stats before cleaning
n_initial = len(df)
n_missing_question = df["question"].isnull().sum()
n_missing_answer = df["answer"].isnull().sum()
n_blank_question = df["question"].dropna().str.strip().eq("").sum()
n_blank_answer = df["answer"].dropna().str.strip().eq("").sum()

# Drop rows with missing or empty question/answer
df = df.dropna(subset=["question", "answer"])
df = df[~df["question"].str.strip().eq("")]
df = df[~df["answer"].str.strip().eq("")]

# Filter by safe question types
safe_types = {
    "causes", "complications", "considerations", "dietary", "exams and tests",
    "genetic changes", "how can i learn more", "how does it work",
    "how effective is it", "information", "inheritance", "outlook",
    "prevention", "research", "stages", "storage and disposal", "support groups",
    "susceptibility", "symptoms", "why get vaccinated"
}
df = df[df["question_type"].str.lower().isin(safe_types)]

# Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
n_train, n_val = int(n * 0.6), int(n * 0.2)

df_train = df[:n_train]
df_val = df[n_train:n_train + n_val]
df_test = df[n_train + n_val:]

# Output directory structure
base_dir = "/data/data/dataset-split"
train_dir = os.path.join(base_dir, "training")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "evaluation")

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Save splits
df_train.to_json(os.path.join(train_dir, "training.json"), orient="records", lines=True)
df_val.to_json(os.path.join(val_dir, "validation.json"), orient="records", lines=True)
df_test.to_json(os.path.join(test_dir, "testing.json"), orient="records", lines=True)

# Split evaluation into multiple production sets
num_production_sets = 3
prod_split_size = len(df_test) // num_production_sets

for i in range(num_production_sets):
    start = i * prod_split_size
    end = None if i == num_production_sets - 1 else (i + 1) * prod_split_size
    df_prod = df_test[start:end]

    prod_dir = os.path.join(base_dir, f"production/batch_{i+1}")
    os.makedirs(prod_dir, exist_ok=True)
    df_prod.to_json(os.path.join(prod_dir, f"prod_batch_{i+1}.json"), orient="records", lines=True)

# Save metadata
metadata = {
    "source": "raw-dataset",
    "initial_records": int(n_initial),
    "final_records": len(df),
    "split_counts": {
        "training": len(df_train),
        "validation": len(df_val),
        "testing": len(df_test)
    },
    "dropped": {
        "missing_question": int(n_missing_question),
        "missing_answer": int(n_missing_answer),
        "blank_question": int(n_blank_question),
        "blank_answer": int(n_blank_answer),
        "total_dropped": int(n_missing_question + n_missing_answer + n_blank_question + n_blank_answer)
    }
}

with open(os.path.join(base_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Final split complete — Train: {len(df_train)}, Val: {len(df_val)}, Eval: {len(df_test)}")
print(f"Production sets created: {num_production_sets}")
