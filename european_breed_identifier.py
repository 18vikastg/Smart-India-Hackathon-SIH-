#!/usr/bin/env python3
"""
European Cattle Breed Identification System
Uses the analyzed breed database to identify new cattle images
"""

import json
from pathlib import Path
from PIL import Image
from datetime import datetime

class EuropeanCattleBreedIdentifier:
    """Identifies cattle breeds against European breed database"""
    
    def __init__(self):
        self.european_breeds = {
            "Jersey": {
                "type": "Dairy Cattle",
                "origin": "Jersey Island", 
                "coat": ["fawn", "light brown", "cream", "yellow-brown"],
                "forehead": ["refined", "dished", "feminine"],
                "ears": ["small", "refined", "alert"],
                "horns": ["small", "curved", "often polled"],
                "build": ["small", "compact", "refined", "dairy type"],
                "hump": ["none"],
                "dewlap": ["minimal"],
                "size": "small",
                "primary_use": "dairy",
                "distinctive_features": [
                    "Fawn to cream colored coat",
                    "Small, refined build",
                    "Highest butterfat content milk",
                    "Dished facial profile",
                    "Efficient feed conversion"
                ]
            },
            "Ayrshire": {
                "type": "Dairy Cattle",
                "origin": "Scotland",
                "coat": ["red", "brown", "white patches", "mahogany and white"],
                "forehead": ["flat", "slightly dished"],
                "ears": ["medium", "alert", "well-set"],
                "horns": ["medium", "curved upward", "often polled"],
                "build": ["medium", "dairy type", "angular", "well-balanced"],
                "hump": ["none"],
                "dewlap": ["minimal", "tight"],
                "size": "medium",
                "primary_use": "dairy", 
                "distinctive_features": [
                    "Red and white patched coat",
                    "Medium-sized, well-balanced build",
                    "Hardy constitution",
                    "Excellent grazing ability",
                    "Good milk quality"
                ]
            },
            "Holstein Friesian": {
                "type": "Dairy Cattle",
                "origin": "Netherlands/Germany",
                "coat": ["black and white patches", "distinctive pattern", "large black patches"],
                "forehead": ["broad", "flat"],
                "ears": ["medium", "alert", "well-positioned"],
                "horns": ["polled", "naturally hornless"],
                "build": ["large", "angular", "dairy type", "tall"],
                "hump": ["none"],
                "dewlap": ["minimal", "tight"],
                "size": "large",
                "primary_use": "dairy",
                "distinctive_features": [
                    "Black and white patched pattern",
                    "Large, angular frame",
                    "Highest milk production",
                    "Naturally polled (hornless)",
                    "Tall, dairy-type conformation"
                ]
            },
            "Brown Swiss": {
                "type": "Dual Purpose Cattle",
                "origin": "Switzerland", 
                "coat": ["brown", "light brown", "dark brown", "solid brown"],
                "forehead": ["broad", "flat"],
                "ears": ["large", "well-set", "alert"],
                "horns": ["short", "curved", "often polled"],
                "build": ["large", "muscular", "sturdy", "dual-purpose"],
                "hump": ["none"],
                "dewlap": ["moderate"],
                "size": "large",
                "primary_use": "dual-purpose",
                "distinctive_features": [
                    "Solid brown coat coloration",
                    "Large, muscular frame",
                    "Black nose and tongue",
                    "Docile temperament", 
                    "Dual-purpose build"
                ]
            },
            "Red Dane": {
                "type": "Dual Purpose Cattle",
                "origin": "Denmark",
                "coat": ["red", "reddish-brown", "solid red", "uniform red"],
                "forehead": ["broad", "flat"],
                "ears": ["medium", "well-set"],
                "horns": ["short", "curved", "often polled"],
                "build": ["medium to large", "dual-purpose", "well-balanced"],
                "hump": ["none"],
                "dewlap": ["moderate"],
                "size": "large",
                "primary_use": "dual-purpose",
                "distinctive_features": [
                    "Uniform red coloration",
                    "Dual-purpose conformation",
                    "Good milk and beef production",
                    "Hardy and adaptable",
                    "Well-balanced build"
                ]
            }
        }
    
    def match_features_to_breed(self, observed_features):
        """Match observed features to European cattle breeds"""
        
        breed_scores = {}
        
        for breed_name, breed_data in self.european_breeds.items():
            score = 0
            max_score = 0
            matches = []
            non_matches = []
            
            # Coat matching (high weight)
            max_score += 3
            coat_match = self.match_feature(observed_features.get("coat", ""), breed_data["coat"])
            if coat_match:
                score += 3
                matches.append(f"Coat: {coat_match}")
            else:
                non_matches.append(f"Coat: {observed_features.get('coat', 'Not specified')} (expected: {', '.join(breed_data['coat'][:2])})")
            
            # Build/Size matching (high weight)
            max_score += 3
            build_match = self.match_feature(observed_features.get("build", ""), breed_data["build"])
            if build_match:
                score += 3 
                matches.append(f"Build: {build_match}")
            else:
                non_matches.append(f"Build: {observed_features.get('build', 'Not specified')} (expected: {', '.join(breed_data['build'][:2])})")
            
            # Forehead matching (medium weight)
            max_score += 2
            forehead_match = self.match_feature(observed_features.get("forehead", ""), breed_data["forehead"])
            if forehead_match:
                score += 2
                matches.append(f"Forehead: {forehead_match}")
            else:
                non_matches.append(f"Forehead: {observed_features.get('forehead', 'Not specified')} (expected: {', '.join(breed_data['forehead'])})")
            
            # Ear matching (medium weight) 
            max_score += 2
            ear_match = self.match_feature(observed_features.get("ears", ""), breed_data["ears"])
            if ear_match:
                score += 2
                matches.append(f"Ears: {ear_match}")
            else:
                non_matches.append(f"Ears: {observed_features.get('ears', 'Not specified')} (expected: {', '.join(breed_data['ears'])})")
            
            # Horn matching (low weight, often polled)
            max_score += 1
            horn_match = self.match_feature(observed_features.get("horns", ""), breed_data["horns"])
            if horn_match:
                score += 1
                matches.append(f"Horns: {horn_match}")
            else:
                non_matches.append(f"Horns: {observed_features.get('horns', 'Not specified')} (expected: {', '.join(breed_data['horns'])})")
            
            confidence = (score / max_score * 100) if max_score > 0 else 0
            
            breed_scores[breed_name] = {
                "score": score,
                "max_score": max_score,
                "confidence": round(confidence, 1),
                "matches": matches,
                "non_matches": non_matches,
                "breed_data": breed_data
            }
        
        # Sort by confidence
        sorted_breeds = sorted(breed_scores.items(), key=lambda x: x[1]["confidence"], reverse=True)
        return sorted_breeds[:3]  # Return top 3 matches
    
    def match_feature(self, observed, expected_list):
        """Check if observed feature matches any expected features"""
        if not observed or not expected_list:
            return None
            
        observed_lower = observed.lower()
        
        for expected in expected_list:
            expected_lower = expected.lower()
            if expected_lower in observed_lower or observed_lower in expected_lower:
                return expected
                
        return None
    
    def identify_cattle_breed(self, observed_features, image_name="Unknown"):
        """Complete breed identification process"""
        
        print(f"\n{'='*60}")
        print(f"EUROPEAN CATTLE BREED IDENTIFICATION")
        print(f"Image: {image_name}")
        print(f"{'='*60}")
        
        # Get top 3 matches
        top_matches = self.match_features_to_breed(observed_features)
        
        print(f"\n## Physical Feature Analysis - {image_name}")
        print(f"\n**Observed Features:**")
        for feature, value in observed_features.items():
            if value:
                print(f"- **{feature.title()}**: {value}")
        
        print(f"\n## Breed Database Comparison")
        print(f"\n**Top 3 European Breed Matches:**")
        
        # Display top 3 matches
        for i, (breed_name, match_data) in enumerate(top_matches, 1):
            breed_info = match_data["breed_data"]
            
            print(f"\n### {i}{'st' if i==1 else 'nd' if i==2 else 'rd'} Match: {breed_name} - {breed_info['type']}")
            print(f"- **Matching Features**: {'; '.join(match_data['matches']) if match_data['matches'] else 'No direct matches'}")
            print(f"- **Non-Matching Features**: {'; '.join(match_data['non_matches'][:2]) if match_data['non_matches'] else 'All features align'}")
            print(f"- **Match Confidence**: {match_data['confidence']}%")
            
            if match_data["matches"]:
                key_evidence = match_data["matches"][0]
            else:
                key_evidence = "Limited feature alignment"
            print(f"- **Key Evidence**: {key_evidence}")
            print(f"- **Origin**: {breed_info['origin']}")
            print(f"- **Primary Use**: {breed_info['primary_use'].title()}")
        
        # Final identification
        if top_matches:
            best_match = top_matches[0]
            breed_name = best_match[0]
            confidence = best_match[1]["confidence"]
            
            print(f"\n## Final Breed Identification")
            print(f"\n### **PREDICTED BREED: {breed_name} ({best_match[1]['breed_data']['type']})**")
            
            print(f"\n### **Primary Reasoning:**")
            strongest_evidence = best_match[1]["matches"][:2] if best_match[1]["matches"] else ["Limited matching features"]
            print(f"- **Strongest Evidence**: {'; '.join(strongest_evidence)}")
            
            supporting_evidence = best_match[1]["matches"][2:] if len(best_match[1]["matches"]) > 2 else []
            if supporting_evidence:
                print(f"- **Supporting Evidence**: {'; '.join(supporting_evidence)}")
            
            distinctive_features = best_match[1]["breed_data"]["distinctive_features"][:2]
            print(f"- **Distinctive Markers**: {'; '.join(distinctive_features)}")
            
            print(f"\n### **Confidence Assessment:**")
            print(f"- **Overall Confidence**: {confidence}%")
            
            if confidence >= 85:
                recommendation = "✅ Classification complete - breed identified with high certainty"
            elif confidence >= 60:
                recommendation = "⚠️ Likely identification - consider additional confirmation"
            else:
                recommendation = "❌ Low confidence - additional information required"
                
            print(f"- **Recommendation**: {recommendation}")
            
            print(f"\n### **Practical Application Notes:**")
            use_type = best_match[1]["breed_data"]["primary_use"]
            if use_type == "dairy":
                print(f"This {breed_name} identification suggests a specialized dairy cattle breed suitable for:")
                print(f"- High-quality milk production")
                print(f"- Intensive dairy management systems")
                print(f"- Temperate climate operations")
            elif use_type == "dual-purpose":
                print(f"This {breed_name} identification suggests a dual-purpose breed suitable for:")
                print(f"- Both milk and beef production")
                print(f"- Versatile farm operations")
                print(f"- Balanced management approach")
            
        return top_matches

def demonstrate_breed_identification():
    """Demonstrate the breed identification system"""
    
    identifier = EuropeanCattleBreedIdentifier()
    
    # Test cases for different breeds
    test_cases = [
        {
            "name": "Test Case 1 - Holstein Pattern",
            "features": {
                "coat": "black and white patches with large irregular markings",
                "build": "large, angular dairy-type frame",
                "forehead": "broad, flat profile", 
                "ears": "medium-sized, alert positioning",
                "horns": "naturally polled, no horns present"
            }
        },
        {
            "name": "Test Case 2 - Jersey Characteristics", 
            "features": {
                "coat": "light fawn color with cream undertones",
                "build": "small, compact, refined dairy build",
                "forehead": "refined, slightly dished profile",
                "ears": "small, refined, alert",
                "horns": "small curved horns"
            }
        },
        {
            "name": "Test Case 3 - Brown Swiss Features",
            "features": {
                "coat": "solid brown coloration throughout",
                "build": "large, muscular, dual-purpose frame", 
                "forehead": "broad, flat profile",
                "ears": "large, well-set ears",
                "horns": "short, curved horns"
            }
        }
    ]
    
    print("=== EUROPEAN CATTLE BREED IDENTIFICATION DEMONSTRATION ===")
    
    for test_case in test_cases:
        results = identifier.identify_cattle_breed(
            test_case["features"], 
            test_case["name"]
        )
        print(f"\n{'-'*60}")
        
    return identifier

if __name__ == "__main__":
    demonstrate_breed_identification()