#!/usr/bin/env python3
"""
Cattle/Buffalo Image Analysis System
Livestock Breed Identification Expert Tool

This script downloads cattle images and performs comprehensive physical feature analysis.
"""

import os
import requests
import json
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
from typing import Dict, List, Optional
import time

class CattleImageAnalyzer:
    """Livestock breed identification expert for physical feature analysis"""
    
    def __init__(self, images_dir: str = "cattle_images"):
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        self.analysis_results = []
        
    def download_sample_images(self) -> List[str]:
        """Download sample cattle images from public sources"""
        
        # Sample cattle images from public sources (these are free-to-use images)
        sample_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Cow_female_black_white.jpg/800px-Cow_female_black_white.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Domestic_water_buffalo_1.jpg/800px-Domestic_water_buffalo_1.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Brahman_bull.jpg/800px-Brahman_bull.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Holstein_cattle_2.jpg/800px-Holstein_cattle_2.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Angus_cattle_2.jpg/800px-Angus_cattle_2.jpg"
        ]
        
        downloaded_files = []
        
        for i, url in enumerate(sample_urls):
            try:
                print(f"Downloading image {i+1}/{len(sample_urls)}...")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                filename = f"cattle_sample_{i+1}.jpg"
                filepath = self.images_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(str(filepath))
                print(f"✓ Downloaded: {filename}")
                time.sleep(1)  # Be respectful to servers
                
            except Exception as e:
                print(f"✗ Failed to download image {i+1}: {e}")
                continue
        
        return downloaded_files
    
    def analyze_image_quality(self, image_path: str) -> Dict[str, str]:
        """Assess image quality for feature visibility"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Basic quality metrics
                resolution = width * height
                aspect_ratio = width / height
                
                # Determine quality rating
                if resolution > 500000:  # > 0.5 MP
                    quality = "Clear - High resolution, good for detailed analysis"
                elif resolution > 200000:  # > 0.2 MP
                    quality = "Good - Moderate resolution, most features visible"
                else:
                    quality = "Limited - Low resolution, some features may be unclear"
                
                return {
                    "resolution": f"{width}x{height}",
                    "megapixels": f"{resolution/1000000:.1f}MP",
                    "aspect_ratio": f"{aspect_ratio:.2f}",
                    "quality_rating": quality
                }
        except Exception as e:
            return {"quality_rating": f"Error analyzing image: {e}"}
    
    def analyze_physical_features(self, image_path: str) -> Dict[str, str]:
        """
        Comprehensive physical feature analysis for livestock breed identification.
        
        Note: This is a template for manual analysis. In a real implementation,
        you would use computer vision models (like YOLO, ResNet, etc.) to detect features.
        """
        
        image_name = Path(image_path).name
        quality_info = self.analyze_image_quality(image_path)
        
        # Template for manual analysis - replace with your observations
        analysis = {
            "image_file": image_name,
            "coat": "Manual observation required - describe exact color, pattern, markings",
            "horns": "Manual observation required - describe horn shape, size, direction, or note if absent",
            "ears": "Manual observation required - describe ear size, positioning, shape",
            "forehead": "Manual observation required - describe profile (flat, convex, dished)",
            "body_structure": "Manual observation required - describe build (compact, lean, muscular)",
            "hump": "Manual observation required - describe hump presence, size, prominence",
            "dewlap": "Manual observation required - describe dewlap characteristics if visible",
            "distinctive_marks": "Manual observation required - note unique spots, patches, markings",
            "image_quality": quality_info["quality_rating"]
        }
        
        return analysis
    
    def format_analysis_report(self, analysis: Dict[str, str]) -> str:
        """Format analysis in the requested structure"""
        
        report = f"""
## Physical Feature Analysis - {analysis['image_file']}

**Coat**: {analysis['coat']}
**Horns**: {analysis['horns']}
**Ears**: {analysis['ears']}
**Forehead**: {analysis['forehead']}
**Body Structure**: {analysis['body_structure']}
**Hump**: {analysis['hump']}
**Dewlap**: {analysis['dewlap']}
**Distinctive Marks**: {analysis['distinctive_marks']}
**Image Quality**: {analysis['image_quality']}

---
"""
        return report
    
    def analyze_all_images(self) -> str:
        """Analyze all downloaded images and generate comprehensive report"""
        
        print("Starting cattle image analysis...")
        
        # Download sample images
        image_files = self.download_sample_images()
        
        if not image_files:
            return "No images were successfully downloaded for analysis."
        
        # Analyze each image
        full_report = "# Cattle/Buffalo Physical Feature Analysis Report\n\n"
        full_report += f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        full_report += f"Total Images Analyzed: {len(image_files)}\n\n"
        
        for image_path in image_files:
            print(f"Analyzing: {Path(image_path).name}")
            analysis = self.analyze_physical_features(image_path)
            self.analysis_results.append(analysis)
            full_report += self.format_analysis_report(analysis)
        
        # Save report
        report_path = self.images_dir / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(full_report)
        
        print(f"\n✓ Analysis complete! Report saved to: {report_path}")
        return full_report

def main():
    """Main function to run the cattle analysis system"""
    
    print("=== Cattle/Buffalo Image Analysis System ===")
    print("Livestock Breed Identification Expert Tool\n")
    
    analyzer = CattleImageAnalyzer()
    
    try:
        # Run the complete analysis
        report = analyzer.analyze_all_images()
        
        print("\n" + "="*60)
        print("MANUAL ANALYSIS REQUIRED")
        print("="*60)
        print("The images have been downloaded to the 'cattle_images' folder.")
        print("Please manually examine each image and update the analysis with specific observations.")
        print("Look for:")
        print("- Exact coat colors and patterns")
        print("- Horn shapes and characteristics") 
        print("- Ear size and positioning")
        print("- Forehead profile")
        print("- Body build and proportions")
        print("- Hump presence and size")
        print("- Dewlap characteristics")
        print("- Any distinctive markings")
        
        return report
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()
