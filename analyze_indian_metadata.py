#!/usr/bin/env python3
"""
Analyze the Indian Bovine Breeds Dataset Metadata
"""

import pandas as pd
import os
from pathlib import Path

def analyze_metadata():
    """Analyze the CSV metadata file"""
    
    dataset_path = "/home/vikas/.cache/kagglehub/datasets/lukex9442/indian-bovine-breeds/versions/5"
    csv_path = os.path.join(dataset_path, "bovine_breeds_metadata.csv")
    
    print("📊 ANALYZING INDIAN BOVINE BREEDS METADATA")
    print("=" * 50)
    
    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"📋 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Analyze breeds
    breed_counts = df['breed'].value_counts()
    print(f"\n🐄 Found {len(breed_counts)} unique breeds:")
    print("\n🏷️  Breed distribution:")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count} images")
    
    # Analyze file extensions
    ext_counts = df['file extension'].value_counts()
    print(f"\n📸 File extensions:")
    for ext, count in ext_counts.items():
        print(f"  .{ext}: {count} files")
    
    # Sample data
    print(f"\n📝 Sample metadata:")
    print(df.head(10).to_string(index=False))
    
    # Check for missing data
    print(f"\n🔍 Missing data check:")
    print(df.isnull().sum())
    
    return df, breed_counts

if __name__ == "__main__":
    df, breeds = analyze_metadata()
    
    print(f"\n✅ Dataset Analysis Complete!")
    print(f"   - Total images: {len(df)}")
    print(f"   - Total breeds: {len(breeds)}")
    print(f"   - Largest breed: {breeds.index[0]} ({breeds.iloc[0]} images)")
    print(f"   - Smallest breed: {breeds.index[-1]} ({breeds.iloc[-1]} images)")