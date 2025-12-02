import os
from datasets import load_dataset
import pandas as pd

# Ensure data/raw dir
os.makedirs('data/raw', exist_ok=True)

# Load dataset from Hugging Face (no auth needed for public datasets)
print("Downloading dataset from Hugging Face...")
dataset = load_dataset("David-Egea/Creditcard-fraud-detection")

# Convert to Pandas (it's a single split, named 'train')
df = dataset['train'].to_pandas()

# Verify shape and columns
print(f"Dataset shape: {df.shape}")  # Should be (284807, 31)
print(f"Columns: {df.columns.tolist()}")  # Time, V1-V28, Amount, Class
print(f"Fraud count: {(df['Class'] == 1).sum()}")  # Should be 492

# Save as CSV
output_path = 'data/raw/creditcard.csv'
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")