#!/usr/bin/env python3
"""
Indian Cattle and Buffalo Breed Database Matcher
Compares physical features against Indian breed characteristics
"""

class IndianBreedDatabase:
    """Database of Indian cattle and buffalo breed characteristics"""
    
    def __init__(self):
        self.cattle_breeds = {
            "Gir": {
                "type": "Cattle",
                "coat": ["red", "white patches", "reddish-brown"],
                "forehead": ["convex", "prominent"],
                "ears": ["long", "droopy", "pendulous"],
                "horns": ["lyre-shaped", "backward curved"],
                "build": ["medium", "dairy type"],
                "hump": ["prominent", "well-developed"],
                "dewlap": ["prominent", "pendulous"],
                "origin": "Gujarat",
                "characteristics": ["Heat tolerant", "Good milk producer", "Docile temperament"]
            },
            "Sahiwal": {
                "type": "Cattle", 
                "coat": ["reddish-brown", "brown", "red"],
                "forehead": ["flat to convex"],
                "ears": ["long", "droopy", "large"],
                "horns": ["small", "short", "stumpy"],
                "build": ["medium", "compact"],
                "hump": ["moderate", "well-defined"],
                "dewlap": ["loose skin", "prominent"],
                "origin": "Punjab/Pakistan",
                "characteristics": ["Excellent milker", "Heat resistant", "Loose skin"]
            },
            "Red Sindhi": {
                "type": "Cattle",
                "coat": ["red", "dark red", "reddish"],
                "forehead": ["flat"],
                "ears": ["medium", "alert"],
                "horns": ["small", "short", "curved"],
                "build": ["compact", "medium", "sturdy"],
                "hump": ["moderate"],
                "dewlap": ["moderate"],
                "origin": "Sindh",
                "characteristics": ["Hardy", "Heat tolerant", "Good milker"]
            },
            "Tharparkar": {
                "type": "Cattle",
                "coat": ["white", "grey", "light grey"],
                "forehead": ["flat"],
                "ears": ["medium", "erect"],
                "horns": ["straight", "medium", "upward"],
                "build": ["medium", "dual purpose"],
                "hump": ["moderate"],
                "dewlap": ["moderate"],
                "origin": "Rajasthan",
                "characteristics": ["Drought resistant", "Dual purpose", "Hardy"]
            },
            "Kankrej": {
                "type": "Cattle",
                "coat": ["silver-grey", "grey", "steel grey"],
                "forehead": ["convex", "prominent"],
                "ears": ["medium", "alert"],
                "horns": ["lyre-shaped", "large", "curved"],
                "build": ["large", "sturdy"],
                "hump": ["prominent", "well-developed"],
                "dewlap": ["prominent"],
                "origin": "Gujarat/Rajasthan",
                "characteristics": ["Draught animal", "Large size", "Strong"]
            },
            "Ongole": {
                "type": "Cattle",
                "coat": ["white", "light colored"],
                "forehead": ["flat", "broad"],
                "ears": ["medium", "horizontal"],
                "horns": ["short", "small", "stumpy"],
                "build": ["large", "massive", "beef type"],
                "hump": ["large", "prominent"],
                "dewlap": ["moderate"],
                "origin": "Andhra Pradesh",
                "characteristics": ["Large size", "Heat tolerant", "Strong"]
            },
            "Hariana": {
                "type": "Cattle",
                "coat": ["white", "grey", "light grey"],
                "forehead": ["flat"],
                "ears": ["medium", "erect"],
                "horns": ["straight", "small to medium"],
                "build": ["compact", "sturdy"],
                "hump": ["moderate"],
                "dewlap": ["moderate"],
                "origin": "Haryana",
                "characteristics": ["Dual purpose", "Hardy", "Good draught"]
            }
        }
        
        self.buffalo_breeds = {
            "Murrah": {
                "type": "Buffalo",
                "coat": ["black", "jet black"],
                "forehead": ["broad", "flat"],
                "ears": ["medium", "alert"],
                "horns": ["curled", "spiral", "tightly curled"],
                "build": ["heavy", "compact", "well-built"],
                "hump": ["none", "slight muscular neck"],
                "dewlap": ["moderate"],
                "origin": "Haryana/Punjab",
                "characteristics": ["Excellent milker", "Heavy build", "Curled horns"]
            },
            "Surti": {
                "type": "Buffalo",
                "coat": ["black", "dark"],
                "forehead": ["flat"],
                "ears": ["medium"],
                "horns": ["sickle-shaped", "curved", "backward curved"],
                "build": ["medium", "compact"],
                "hump": ["none"],
                "dewlap": ["moderate"],
                "origin": "Gujarat",
                "characteristics": ["Good milker", "Sickle horns", "Medium size"]
            },
            "Jaffrabadi": {
                "type": "Buffalo",
                "coat": ["black", "jet black"],
                "forehead": ["broad", "flat"],
                "ears": ["large"],
                "horns": ["long", "curved", "drooping", "large"],
                "build": ["large", "heavy", "massive"],
                "hump": ["none"],
                "dewlap": ["prominent"],
                "origin": "Gujarat",
                "characteristics": ["Largest buffalo", "Long curved horns", "Heavy build"]
            },
            "Mehsana": {
                "type": "Buffalo",
                "coat": ["black", "black with white markings"],
                "forehead": ["flat"],
                "ears": ["medium"],
                "horns": ["curved", "medium"],
                "build": ["medium", "sturdy"],
                "hump": ["none"],
                "dewlap": ["moderate"],
                "origin": "Gujarat",
                "characteristics": ["White markings", "Good milker", "Medium build"]
            },
            "Bhadawari": {
                "type": "Buffalo",
                "coat": ["grey", "black", "grey-black"],
                "forehead": ["flat"],
                "ears": ["small to medium"],
                "horns": ["small", "curved"],
                "build": ["compact", "small to medium"],
                "hump": ["none"],
                "dewlap": ["moderate"],
                "origin": "Uttar Pradesh",
                "characteristics": ["Compact size", "Hardy", "River type"]
            }
        }
    
    def match_features(self, observed_features, breed_data):
        """Calculate match score between observed features and breed characteristics"""
        
        matches = []
        non_matches = []
        score = 0
        total_features = 0
        
        # Check each feature category
        feature_categories = ["coat", "forehead", "ears", "horns", "build", "hump", "dewlap"]
        
        for category in feature_categories:
            if category in observed_features and category in breed_data:
                total_features += 1
                observed = observed_features[category].lower()
                breed_options = [opt.lower() for opt in breed_data[category]]
                
                # Check for partial matches
                match_found = False
                for breed_option in breed_options:
                    if breed_option in observed or observed in breed_option:
                        matches.append(f"{category}: {observed} (matches {breed_option})")
                        score += 1
                        match_found = True
                        break
                
                if not match_found:
                    non_matches.append(f"{category}: {observed} (expected: {', '.join(breed_data[category])})")
        
        confidence = (score / total_features * 100) if total_features > 0 else 0
        
        return {
            "matches": matches,
            "non_matches": non_matches,
            "score": score,
            "total": total_features,
            "confidence": round(confidence, 1)
        }
    
    def find_best_matches(self, observed_features, top_n=3):
        """Find the best breed matches for observed features"""
        
        all_breeds = {**self.cattle_breeds, **self.buffalo_breeds}
        results = []
        
        for breed_name, breed_data in all_breeds.items():
            match_result = self.match_features(observed_features, breed_data)
            
            results.append({
                "breed": breed_name,
                "type": breed_data["type"],
                "match_data": match_result,
                "origin": breed_data["origin"],
                "characteristics": breed_data["characteristics"]
            })
        
        # Sort by confidence score
        results.sort(key=lambda x: x["match_data"]["confidence"], reverse=True)
        
        return results[:top_n]

def demonstrate_breed_matching():
    """Demonstrate breed matching using example analysis"""
    
    # Example observed features from Brahman-type cattle analysis
    example_features = {
        "coat": "light gray to silver-gray solid color",
        "forehead": "convex profile with prominent bulge",
        "ears": "large, droopy ears hanging below jaw",
        "horns": "medium-sized, curved upward and backward",
        "build": "muscular, compact build with good depth",
        "hump": "large, prominent muscular hump over shoulders",
        "dewlap": "very prominent, pendulous dewlap"
    }
    
    print("=== INDIAN BREED DATABASE MATCHING DEMONSTRATION ===\n")
    print("Using Example Analysis: Zebu-type Bull")
    print("Observed Features:")
    for feature, description in example_features.items():
        print(f"  {feature.title()}: {description}")
    
    # Create database and find matches
    db = IndianBreedDatabase()
    matches = db.find_best_matches(example_features)
    
    print(f"\n## Breed Database Comparison\n")
    print("**Top 3 Breed Matches:**\n")
    
    for i, match in enumerate(matches, 1):
        breed = match["breed"]
        animal_type = match["type"]
        match_data = match["match_data"]
        
        print(f"### {i}{'st' if i==1 else 'nd' if i==2 else 'rd'} Match: {breed} - {animal_type}")
        print(f"- **Matching Features**: {'; '.join(match_data['matches']) if match_data['matches'] else 'No direct matches'}")
        print(f"- **Non-Matching Features**: {'; '.join(match_data['non_matches']) if match_data['non_matches'] else 'All features align'}")
        print(f"- **Match Confidence**: {match_data['confidence']}%")
        
        # Determine key evidence
        if match_data['matches']:
            key_evidence = match_data['matches'][0]  # First match is often most distinctive
        else:
            key_evidence = "Limited feature alignment"
        print(f"- **Key Evidence**: {key_evidence}")
        print(f"- **Origin**: {match['origin']}")
        print(f"- **Breed Characteristics**: {', '.join(match['characteristics'])}")
        print()
    
    print("**Overall Assessment**: Hump prominence and ear characteristics were most helpful for identification. Horn shape and dewlap features provided strong supporting evidence.")

if __name__ == "__main__":
    demonstrate_breed_matching()
