import os
from datasets import load_dataset

print("Downloading dataset 'phreshphish/phreshphish'...")
ds = load_dataset("phreshphish/phreshphish")
print("\nDataset loaded successfully:")
print(ds)

# Save to disk in Hugging Face Arrow format
save_dir = os.path.join(os.getcwd(), "phreshphish_data")
os.makedirs(save_dir, exist_ok=True)
ds.save_to_disk(save_dir)
print(f"\nSaved Hugging Face dataset to: {save_dir}")

# Also export splits to CSV and Parquet for convenience
for split_name in ds.keys():
    split_data = ds[split_name]
    csv_path = os.path.join(save_dir, f"{split_name}.csv")
    parquet_path = os.path.join(save_dir, f"{split_name}.parquet")
    
    print(f"Exporting '{split_name}' ({len(split_data)} rows) to CSV and Parquet...")
    split_data.to_csv(csv_path)
    split_data.to_parquet(parquet_path)
    print(f"  -> CSV: {csv_path}")
    print(f"  -> Parquet: {parquet_path}")

print("\nAll downloads and exports completed successfully!")
