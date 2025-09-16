#!/usr/bin/env python3
"""
Alternative Cattle Image Downloader
Uses different sources and methods to obtain cattle images for analysis
"""

import os
import requests
import json
from pathlib import Path
from PIL import Image
import time
from typing import List

class AlternativeCattleImageDownloader:
    """Alternative method to download cattle images for analysis"""
    
    def __init__(self, images_dir: str = "cattle_images"):
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
    def download_from_unsplash(self) -> List[str]:
        """Download cattle images from Unsplash API (free tier)"""
        
        # Unsplash provides free API access for development
        # Note: In production, you'd need an API key
        
        base_url = "https://source.unsplash.com"
        search_terms = ["cow", "cattle", "bull", "buffalo", "livestock"]
        downloaded_files = []
        
        for i, term in enumerate(search_terms):
            try:
                # Unsplash source URL format
                url = f"{base_url}/800x600/?{term}"
                
                print(f"Downloading {term} image...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                filename = f"sample_{term}_{i+1}.jpg"
                filepath = self.images_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(str(filepath))
                print(f"✓ Downloaded: {filename}")
                time.sleep(2)  # Be respectful to servers
                
            except Exception as e:
                print(f"✗ Failed to download {term} image: {e}")
                continue
        
        return downloaded_files
    
    def create_sample_dataset_urls(self):
        """Create a list of alternative free cattle image sources"""
        
        urls_file = self.images_dir / "cattle_image_sources.txt"
        
        sources = """
# Free Cattle Image Sources for Manual Download

## Pixabay (Free for commercial use)
https://pixabay.com/images/search/cattle/
https://pixabay.com/images/search/cow/
https://pixabay.com/images/search/bull/
https://pixabay.com/images/search/buffalo/

## Pexels (Free stock photos)
https://www.pexels.com/search/cattle/
https://www.pexels.com/search/cow/
https://www.pexels.com/search/livestock/

## Unsplash (Free high-resolution photos)
https://unsplash.com/s/photos/cattle
https://unsplash.com/s/photos/cow
https://unsplash.com/s/photos/bull

## Government Agricultural Sources
https://www.ars.usda.gov/ (USDA Agricultural Research Service)
https://www.fao.org/ (FAO Animal Genetic Resources)

## Instructions:
1. Visit these websites
2. Search for cattle/buffalo images
3. Download 5-10 high-quality images
4. Save them in the 'cattle_images' folder
5. Run the analysis script to examine them

## Recommended Image Criteria:
- High resolution (at least 800x600)
- Clear side or front view of the animal
- Good lighting showing coat patterns
- Visible facial features and body structure
- Different breeds if possible
"""
        
        with open(urls_file, 'w') as f:
            f.write(sources)
        
        print(f"✓ Created image source guide: {urls_file}")
        return str(urls_file)

def main():
    print("=== Alternative Cattle Image Downloader ===\n")
    
    downloader = AlternativeCattleImageDownloader()
    
    # Try Unsplash method
    print("Attempting to download from Unsplash...")
    downloaded = downloader.download_from_unsplash()
    
    if downloaded:
        print(f"\n✓ Successfully downloaded {len(downloaded)} images!")
        print("Images saved in 'cattle_images' folder")
    else:
        print("\n⚠ Automatic download failed. Creating manual download guide...")
        guide_file = downloader.create_sample_dataset_urls()
        print(f"Please see {guide_file} for manual download instructions.")
    
    print("\nNext steps:")
    print("1. Ensure you have cattle images in the 'cattle_images' folder")
    print("2. Run 'python cattle_analyzer.py' to start the analysis")
    print("3. Use 'analysis_template.md' as your guide for manual feature detection")

if __name__ == "__main__":
    main()
