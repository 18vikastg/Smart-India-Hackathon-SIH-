#!/usr/bin/env python3
"""
Interactive Cattle Physical Feature Analysis Tool
Livestock Breed Identification Expert Assistant
"""

import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

class InteractiveCattleAnalyzer:
    """Interactive tool for detailed cattle physical feature analysis"""
    
    def __init__(self):
        self.images_dir = Path("cattle_images")
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def get_available_images(self):
        """Get list of image files in the cattle_images directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        if self.images_dir.exists():
            for file in self.images_dir.iterdir():
                if file.suffix.lower() in image_extensions:
                    images.append(file)
        
        return sorted(images)
    
    def display_image_info(self, image_path):
        """Display basic information about an image"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"\nImage: {image_path.name}")
                print(f"Dimensions: {width}x{height}")
                print(f"Format: {img.format}")
                print(f"Mode: {img.mode}")
                
                # Calculate quality metrics
                resolution = width * height
                if resolution > 500000:
                    quality = "High"
                elif resolution > 200000:
                    quality = "Medium" 
                else:
                    quality = "Low"
                    
                print(f"Resolution Quality: {quality}")
                return True
        except Exception as e:
            print(f"Error reading image: {e}")
            return False
    
    def get_user_input(self, prompt, options=None):
        """Get user input with optional validation"""
        while True:
            try:
                response = input(f"{prompt}: ").strip()
                if options and response.lower() not in [opt.lower() for opt in options]:
                    print(f"Please choose from: {', '.join(options)}")
                    continue
                return response
            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user.")
                return None
    
    def analyze_single_image(self, image_path):
        """Interactive analysis of a single cattle image"""
        
        print("\n" + "="*60)
        print(f"ANALYZING: {image_path.name}")
        print("="*60)
        
        if not self.display_image_info(image_path):
            return None
        
        print(f"\nPlease examine the image: {image_path}")
        print("Open the image file to view it while answering the following questions.")
        
        # Collect analysis data interactively
        analysis = {
            "image_file": image_path.name,
            "analysis_date": datetime.now().isoformat(),
            "image_path": str(image_path)
        }
        
        # Coat analysis
        print("\n--- COAT ANALYSIS ---")
        analysis["coat_primary_color"] = self.get_user_input(
            "Primary coat color (e.g., black, white, brown, red, gray)"
        )
        
        analysis["coat_pattern"] = self.get_user_input(
            "Coat pattern",
            ["solid", "spotted", "patched", "brindled", "roan", "mixed"]
        )
        
        if analysis["coat_pattern"] != "solid":
            analysis["coat_pattern_details"] = self.get_user_input(
                "Describe the pattern details (locations, secondary colors, etc.)"
            )
        
        analysis["distinctive_markings"] = self.get_user_input(
            "Any distinctive markings? (white blaze, stockings, patches, etc.)"
        )
        
        # Horn analysis
        print("\n--- HORN ANALYSIS ---")
        analysis["horns_present"] = self.get_user_input(
            "Are horns present?",
            ["yes", "no", "polled"]
        )
        
        if analysis["horns_present"].lower() == "yes":
            analysis["horn_shape"] = self.get_user_input(
                "Horn shape",
                ["straight", "curved", "lyre-shaped", "twisted", "crescent"]
            )
            analysis["horn_size"] = self.get_user_input(
                "Horn size",
                ["small", "medium", "large"]
            )
            analysis["horn_direction"] = self.get_user_input(
                "Horn direction (forward, backward, upward, outward, etc.)"
            )
        
        # Ear analysis
        print("\n--- EAR ANALYSIS ---")
        analysis["ear_size"] = self.get_user_input(
            "Ear size relative to head",
            ["small", "medium", "large"]
        )
        analysis["ear_position"] = self.get_user_input(
            "Ear position",
            ["erect", "semi-erect", "droopy", "horizontal"]
        )
        analysis["ear_shape"] = self.get_user_input(
            "Ear shape (pointed, rounded, broad, narrow, etc.)"
        )
        
        # Head and face analysis
        print("\n--- HEAD & FACE ANALYSIS ---")
        analysis["forehead_profile"] = self.get_user_input(
            "Forehead profile",
            ["flat", "convex", "dished", "bulging"]
        )
        analysis["face_markings"] = self.get_user_input(
            "Facial markings or distinctive features"
        )
        
        # Body structure analysis
        print("\n--- BODY STRUCTURE ANALYSIS ---")
        analysis["body_build"] = self.get_user_input(
            "Overall body build",
            ["compact", "lean", "muscular", "angular", "stocky"]
        )
        analysis["frame_size"] = self.get_user_input(
            "Frame size",
            ["small", "medium", "large"]
        )
        analysis["muscle_definition"] = self.get_user_input(
            "Muscle definition",
            ["minimal", "moderate", "well-defined", "heavily-muscled"]
        )
        
        # Hump analysis
        print("\n--- HUMP ANALYSIS ---")
        analysis["hump_present"] = self.get_user_input(
            "Is a hump present?",
            ["yes", "no", "slight"]
        )
        
        if analysis["hump_present"].lower() in ["yes", "slight"]:
            analysis["hump_size"] = self.get_user_input(
                "Hump size",
                ["small", "medium", "large", "prominent"]
            )
            analysis["hump_location"] = self.get_user_input(
                "Hump location (shoulders, neck, back, etc.)"
            )
        
        # Dewlap analysis
        print("\n--- DEWLAP ANALYSIS ---")
        analysis["dewlap_prominence"] = self.get_user_input(
            "Dewlap prominence",
            ["none", "minimal", "moderate", "prominent", "pendulous"]
        )
        
        if analysis["dewlap_prominence"] not in ["none", "minimal"]:
            analysis["dewlap_extent"] = self.get_user_input(
                "How far does the dewlap extend? (throat, chest, between legs, etc.)"
            )
        
        # Additional features
        print("\n--- ADDITIONAL FEATURES ---")
        analysis["tail_switch_color"] = self.get_user_input(
            "Tail switch color (if visible)"
        )
        analysis["muzzle_pigmentation"] = self.get_user_input(
            "Muzzle pigmentation (black, pink, mixed, etc.)"
        )
        analysis["other_distinctive_features"] = self.get_user_input(
            "Any other distinctive features or breed characteristics?"
        )
        
        # Image quality assessment
        print("\n--- IMAGE QUALITY ASSESSMENT ---")
        analysis["visibility_rating"] = self.get_user_input(
            "How clearly can you see the key features?",
            ["excellent", "good", "fair", "poor"]
        )
        analysis["limiting_factors"] = self.get_user_input(
            "Any factors limiting analysis? (angle, lighting, resolution, etc.)"
        )
        
        return analysis
    
    def format_analysis_report(self, analysis):
        """Format the analysis into the requested report structure"""
        
        # Build coat description
        coat_desc = analysis["coat_primary_color"]
        if analysis["coat_pattern"] != "solid":
            coat_desc += f" with {analysis['coat_pattern']} pattern"
            if "coat_pattern_details" in analysis:
                coat_desc += f" - {analysis['coat_pattern_details']}"
        
        # Build horn description
        if analysis["horns_present"].lower() == "no" or analysis["horns_present"].lower() == "polled":
            horn_desc = "Polled (no horns present)"
        else:
            horn_desc = f"{analysis['horn_size']} {analysis['horn_shape']} horns"
            horn_desc += f", directed {analysis['horn_direction']}"
        
        # Build hump description
        if analysis["hump_present"].lower() == "no":
            hump_desc = "No hump present"
        else:
            hump_desc = f"{analysis['hump_size']} hump over {analysis['hump_location']}"
        
        report = f"""
## Physical Feature Analysis - {analysis['image_file']}

**Coat**: {coat_desc}
**Horns**: {horn_desc}
**Ears**: {analysis['ear_size']}, {analysis['ear_position']}, {analysis['ear_shape']}
**Forehead**: {analysis['forehead_profile']} profile
**Body Structure**: {analysis['body_build']} build, {analysis['frame_size']} frame, {analysis['muscle_definition']} musculature
**Hump**: {hump_desc}
**Dewlap**: {analysis['dewlap_prominence']} dewlap{f" extending to {analysis.get('dewlap_extent', '')}" if analysis['dewlap_prominence'] not in ['none', 'minimal'] else ""}
**Distinctive Marks**: {analysis['distinctive_markings']} | Face: {analysis['face_markings']} | Other: {analysis['other_distinctive_features']}
**Image Quality**: {analysis['visibility_rating'].title()} - {analysis.get('limiting_factors', 'No major limitations')}

**Additional Details:**
- Tail switch: {analysis['tail_switch_color']}
- Muzzle: {analysis['muzzle_pigmentation']}
- Analysis date: {analysis['analysis_date']}

---
"""
        return report
    
    def save_analysis(self, analysis, report):
        """Save analysis data and report"""
        
        # Save detailed JSON data
        json_file = self.results_dir / f"{analysis['image_file']}_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save formatted report
        report_file = self.results_dir / f"{analysis['image_file']}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Analysis saved:")
        print(f"  Data: {json_file}")
        print(f"  Report: {report_file}")
    
    def run_interactive_analysis(self):
        """Main interactive analysis workflow"""
        
        print("="*60)
        print("INTERACTIVE CATTLE PHYSICAL FEATURE ANALYZER")
        print("Livestock Breed Identification Expert Tool")
        print("="*60)
        
        # Check for images
        images = self.get_available_images()
        
        if not images:
            print(f"\nNo images found in {self.images_dir}")
            print("Please add cattle/buffalo images to the 'cattle_images' folder and try again.")
            print("\nSupported formats: JPG, JPEG, PNG, BMP, TIFF")
            return
        
        print(f"\nFound {len(images)} image(s) for analysis:")
        for i, img in enumerate(images, 1):
            print(f"  {i}. {img.name}")
        
        # Process each image
        for image_path in images:
            try:
                print(f"\n{'='*60}")
                response = self.get_user_input(
                    f"Analyze {image_path.name}? (y/n/q for quit)",
                    ["y", "yes", "n", "no", "q", "quit"]
                )
                
                if response and response.lower() in ["q", "quit"]:
                    print("Analysis session ended by user.")
                    break
                elif response and response.lower() in ["n", "no"]:
                    print(f"Skipping {image_path.name}")
                    continue
                
                # Perform the analysis
                analysis = self.analyze_single_image(image_path)
                
                if analysis:
                    # Generate and save report
                    report = self.format_analysis_report(analysis)
                    self.save_analysis(analysis, report)
                    
                    print(f"\n{report}")
                
            except KeyboardInterrupt:
                print(f"\nAnalysis interrupted. Partial results may be saved.")
                break
            except Exception as e:
                print(f"Error analyzing {image_path.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("ANALYSIS SESSION COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print("="*60)

def main():
    """Main function"""
    analyzer = InteractiveCattleAnalyzer()
    analyzer.run_interactive_analysis()

if __name__ == "__main__":
    main()
