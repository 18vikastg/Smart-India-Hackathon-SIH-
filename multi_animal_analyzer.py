#!/usr/bin/env python3
"""
Multi-Animal Breed Analysis Demonstration
Shows comparative livestock identification across multiple animals
"""

from breed_matcher import IndianBreedDatabase

class MultiAnimalAnalyzer:
    """Analyzes and compares multiple cattle/buffalo for breed identification"""
    
    def __init__(self):
        self.breed_db = IndianBreedDatabase()
        self.animals_analyzed = []
    
    def analyze_animal_group(self, animal_features_list):
        """Analyze multiple animals and perform comparative assessment"""
        
        print("="*80)
        print("MULTI-ANIMAL BREED ANALYSIS")
        print("="*80)
        
        # Individual analysis for each animal
        individual_results = []
        
        for i, features in enumerate(animal_features_list, 1):
            print(f"\n--- ANALYZING ANIMAL {i} ---")
            matches = self.breed_db.find_best_matches(features, top_n=3)
            
            # Get top prediction
            top_match = matches[0] if matches else None
            
            result = {
                "animal_id": i,
                "features": features,
                "top_prediction": top_match,
                "all_matches": matches
            }
            
            individual_results.append(result)
            
            if top_match:
                print(f"Animal {i} - Predicted: {top_match['breed']} ({top_match['type']}) - {top_match['match_data']['confidence']}%")
            else:
                print(f"Animal {i} - No clear breed match found")
        
        # Perform comparative analysis
        self.perform_comparative_analysis(individual_results)
        
        return individual_results
    
    def perform_comparative_analysis(self, results):
        """Compare animals and identify patterns"""
        
        print(f"\n{'='*80}")
        print("## Multi-Animal Breed Analysis")
        print(f"{'='*80}")
        
        # Individual Breed Predictions
        print("\n### Individual Breed Predictions:")
        for result in results:
            animal_id = result["animal_id"]
            top_pred = result["top_prediction"]
            
            if top_pred:
                breed = top_pred["breed"]
                animal_type = top_pred["type"]
                confidence = top_pred["match_data"]["confidence"]
                
                # Get key features summary
                key_features = []
                if result["features"].get("coat"):
                    key_features.append(f"coat: {result['features']['coat'][:30]}...")
                if result["features"].get("horns"):
                    key_features.append(f"horns: {result['features']['horns'][:30]}...")
                if result["features"].get("hump"):
                    key_features.append(f"hump: {result['features']['hump'][:30]}...")
                
                print(f"- **Animal {animal_id}**: {breed} ({animal_type}), {confidence}% confidence")
                print(f"  - Key features: {'; '.join(key_features[:3])}")
            else:
                print(f"- **Animal {animal_id}**: Unidentified, insufficient matching features")
        
        # Comparative Analysis
        print(f"\n### Comparative Analysis:")
        
        # Check for similarities
        similarities = self.find_common_features(results)
        print(f"- **Similarities**: {similarities}")
        
        # Check for differences  
        differences = self.find_distinguishing_features(results)
        print(f"- **Differences**: {differences}")
        
        # Group consistency
        consistency = self.assess_group_consistency(results)
        print(f"- **Group Consistency**: {consistency}")
        
        # Identify outliers
        outliers = self.identify_outliers(results)
        print(f"- **Outliers**: {outliers}")
        
        # Recommendations
        recommendations = self.generate_recommendations(results)
        print(f"\n### Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    
    def find_common_features(self, results):
        """Identify features common across animals"""
        if len(results) < 2:
            return "Insufficient animals for comparison"
        
        common_features = []
        
        # Check for common breed types
        animal_types = [r["top_prediction"]["type"] for r in results if r["top_prediction"]]
        if len(set(animal_types)) == 1 and animal_types:
            common_features.append(f"All are {animal_types[0].lower()}")
        
        # Check for common physical traits
        feature_categories = ["coat", "horns", "hump", "ears"]
        for category in feature_categories:
            values = []
            for result in results:
                if category in result["features"]:
                    # Extract key descriptors
                    feature_text = result["features"][category].lower()
                    if "prominent" in feature_text or "large" in feature_text:
                        values.append("prominent/large")
                    elif "small" in feature_text or "minimal" in feature_text:
                        values.append("small/minimal")
                    elif "medium" in feature_text or "moderate" in feature_text:
                        values.append("medium/moderate")
            
            if len(set(values)) == 1 and values:
                common_features.append(f"{category}: {values[0]} across all animals")
        
        return "; ".join(common_features) if common_features else "No consistent features across all animals"
    
    def find_distinguishing_features(self, results):
        """Identify features that distinguish animals from each other"""
        if len(results) < 2:
            return "Single animal - no comparisons possible"
        
        differences = []
        
        # Compare breeds
        breeds = [r["top_prediction"]["breed"] for r in results if r["top_prediction"]]
        if len(set(breeds)) > 1:
            breed_summary = ", ".join(set(breeds))
            differences.append(f"Mixed breeds present: {breed_summary}")
        
        # Compare confidence levels
        confidences = [r["top_prediction"]["match_data"]["confidence"] for r in results if r["top_prediction"]]
        if confidences:
            conf_range = f"{min(confidences):.1f}%-{max(confidences):.1f}%"
            differences.append(f"Identification confidence varies: {conf_range}")
        
        return "; ".join(differences) if differences else "Animals show consistent characteristics"
    
    def assess_group_consistency(self, results):
        """Assess whether animals appear to be from consistent breeding"""
        valid_results = [r for r in results if r["top_prediction"]]
        
        if len(valid_results) < 2:
            return "Insufficient identifiable animals for consistency assessment"
        
        # Check breed consistency
        breeds = [r["top_prediction"]["breed"] for r in valid_results]
        unique_breeds = set(breeds)
        
        if len(unique_breeds) == 1:
            return f"High consistency - all animals identified as {breeds[0]}"
        elif len(unique_breeds) <= len(valid_results) / 2:
            return f"Moderate consistency - {len(unique_breeds)} breeds among {len(valid_results)} animals"
        else:
            return f"Low consistency - high breed diversity ({len(unique_breeds)} breeds among {len(valid_results)} animals)"
    
    def identify_outliers(self, results):
        """Identify animals that differ significantly from the group"""
        if len(results) < 3:
            return "Too few animals to identify outliers"
        
        valid_results = [r for r in results if r["top_prediction"]]
        
        if len(valid_results) < 3:
            return "Insufficient identifiable animals to determine outliers"
        
        # Find most common breed
        breeds = [r["top_prediction"]["breed"] for r in valid_results]
        breed_counts = {}
        for breed in breeds:
            breed_counts[breed] = breed_counts.get(breed, 0) + 1
        
        most_common_breed = max(breed_counts, key=breed_counts.get)
        outliers = []
        
        for result in valid_results:
            if result["top_prediction"]["breed"] != most_common_breed:
                animal_id = result["animal_id"]
                breed = result["top_prediction"]["breed"]
                outliers.append(f"Animal {animal_id} ({breed})")
        
        return "; ".join(outliers) if outliers else f"No significant outliers - group centers around {most_common_breed}"
    
    def generate_recommendations(self, results):
        """Generate herd management recommendations"""
        recommendations = []
        valid_results = [r for r in results if r["top_prediction"]]
        
        if not valid_results:
            return ["Unable to generate recommendations - no animals successfully identified"]
        
        # Breed diversity recommendations
        breeds = [r["top_prediction"]["breed"] for r in valid_results]
        unique_breeds = set(breeds)
        
        if len(unique_breeds) == 1:
            breed = breeds[0]
            recommendations.append(f"Uniform herd of {breed} cattle - maintain consistent breeding program")
            recommendations.append(f"Focus on {breed}-specific management practices and nutrition")
        else:
            recommendations.append(f"Mixed breed herd detected - consider grouping by breed for specialized management")
            recommendations.append("Evaluate crossbreeding potential for improved hybrid vigor")
        
        # Confidence-based recommendations
        confidences = [r["top_prediction"]["match_data"]["confidence"] for r in valid_results]
        low_confidence_animals = [r["animal_id"] for r in valid_results 
                                if r["top_prediction"]["match_data"]["confidence"] < 60]
        
        if low_confidence_animals:
            animal_list = ", ".join([f"Animal {aid}" for aid in low_confidence_animals])
            recommendations.append(f"Obtain additional photos/information for {animal_list} - uncertain identification")
        
        # Management recommendations
        animal_types = [r["top_prediction"]["type"] for r in valid_results]
        if "Buffalo" in animal_types and "Cattle" in animal_types:
            recommendations.append("Mixed cattle/buffalo herd - ensure separate management protocols for each species")
        
        return recommendations

def demonstrate_multi_animal_analysis():
    """Demonstrate multi-animal analysis with example data"""
    
    # Example data for 4 different animals
    animal_data = [
        {
            "animal_name": "Large Black Bull",
            "features": {
                "coat": "solid black coat throughout",
                "forehead": "convex profile with prominent bulge",
                "ears": "large, droopy ears hanging below jaw",
                "horns": "medium-sized, curved upward and backward",
                "build": "large, muscular build with heavy frame",
                "hump": "very prominent muscular hump over shoulders",
                "dewlap": "very prominent, pendulous dewlap"
            }
        },
        {
            "animal_name": "Red and White Cow",
            "features": {
                "coat": "red with white patches on face and legs",
                "forehead": "convex profile, moderately prominent",
                "ears": "long, droopy ears extending below jaw",
                "horns": "lyre-shaped, medium-sized, curved outward",
                "build": "medium, dairy-type build",
                "hump": "moderate hump over shoulders",
                "dewlap": "prominent dewlap extending to chest"
            }
        },
        {
            "animal_name": "Gray Compact Animal",
            "features": {
                "coat": "silver-gray solid color",
                "forehead": "convex profile with pronounced bulge",
                "ears": "medium-sized, alert positioning",
                "horns": "lyre-shaped, large, curved backward",
                "build": "large, sturdy draught-type build",
                "hump": "very prominent hump over shoulders",
                "dewlap": "prominent dewlap"
            }
        },
        {
            "animal_name": "Small Black Buffalo",
            "features": {
                "coat": "jet black solid color",
                "forehead": "broad, flat profile",
                "ears": "medium-sized, horizontal positioning",
                "horns": "tightly curled, spiral-shaped",
                "build": "compact, heavy build",
                "hump": "none - muscular neck region only",
                "dewlap": "moderate dewlap"
            }
        }
    ]
    
    analyzer = MultiAnimalAnalyzer()
    
    print("DEMONSTRATION: Multi-Animal Breed Analysis")
    print("Analyzing 4 different livestock animals...\n")
    
    # Extract just the features for analysis
    features_list = [animal["features"] for animal in animal_data]
    
    # Perform the analysis
    results = analyzer.analyze_animal_group(features_list)
    
    return results

if __name__ == "__main__":
    demonstrate_multi_animal_analysis()
