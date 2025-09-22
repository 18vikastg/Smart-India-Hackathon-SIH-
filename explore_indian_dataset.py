#!/usr/bin/env python3
"""
Explore the Indian Bovine Breeds Dataset from Kaggle
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path

def explore_indian_dataset():
    """Download and explore the Indian bovine breeds dataset"""
    
    print("🐄 EXPLORING INDIAN BOVINE BREEDS DATASET")
    print("=" * 50)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("lukex9442/indian-bovine-breeds")
        print(f"✅ Dataset downloaded to: {dataset_path}")
        
        # List all files in the dataset
        print("\n📁 Dataset Contents:")
        dataset_dir = Path(dataset_path)
        all_files = list(dataset_dir.rglob("*"))
        
        for file_path in sorted(all_files)[:20]:  # Show first 20 files
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"  📄 {file_path.relative_to(dataset_dir)} ({size_mb:.2f} MB)")
        
        if len(all_files) > 20:
            print(f"  ... and {len(all_files) - 20} more files")
        
        # Look for CSV files that might contain labels
        csv_files = list(dataset_dir.rglob("*.csv"))
        print(f"\n📊 Found {len(csv_files)} CSV files:")
        
        for csv_file in csv_files:
            print(f"  📋 {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                print(f"    - Shape: {df.shape}")
                print(f"    - Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"    - Sample data:")
                    print(f"      {df.head(2).to_dict('records')}")
                print()
            except Exception as e:
                print(f"    - Error reading CSV: {e}")
        
        # Look for image folders
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        breed_folders = {}
        
        for file_path in all_files:
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
                # Check if organized by breed folders
                parent_name = file_path.parent.name
                if parent_name not in breed_folders:
                    breed_folders[parent_name] = 0
                breed_folders[parent_name] += 1
        
        print(f"🖼️  Found {len(image_files)} image files")
        
        if breed_folders:
            print(f"\n🏷️  Potential breed folders:")
            for folder, count in sorted(breed_folders.items(), key=lambda x: x[1], reverse=True):
                if count > 1:  # Only show folders with multiple images
                    print(f"  📁 {folder}: {count} images")
        
        # Sample a few image paths
        if image_files:
            print(f"\n📸 Sample image paths:")
            for img_path in image_files[:5]:
                print(f"  🖼️  {img_path.relative_to(dataset_dir)}")
        
        return dataset_path, image_files, breed_folders, csv_files
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        return None, [], {}, []

if __name__ == "__main__":
    dataset_path, images, breeds, csvs = explore_indian_dataset()
    
    print(f"\n📋 SUMMARY:")
    print(f"  📁 Dataset Path: {dataset_path}")
    print(f"  🖼️  Total Images: {len(images)}")
    print(f"  🏷️  Potential Breeds: {len([b for b, c in breeds.items() if c > 1])}")
    print(f"  📊 CSV Files: {len(csvs)}")