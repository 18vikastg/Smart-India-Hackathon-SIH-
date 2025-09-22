# Health Assessment and Breeding Insights for Cattle Recognition
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os
from dataclasses import dataclass
from enum import Enum


class BodyConditionScale(Enum):
    """Body Condition Score scale (1-9)"""
    EMACIATED = 1
    VERY_THIN = 2
    THIN = 3
    MODERATELY_THIN = 4
    MODERATE = 5
    MODERATELY_FLESHY = 6
    FLESHY = 7
    FAT = 8
    VERY_FAT = 9


@dataclass
class HealthAssessment:
    """Data class for health assessment results"""
    body_condition_score: float
    condition_category: str
    health_status: str
    weight_estimate_kg: Optional[float] = None
    age_estimate_months: Optional[int] = None
    reproductive_status: Optional[str] = None


@dataclass
class BreedingRecommendation:
    """Data class for breeding recommendations"""
    breeding_readiness: str
    optimal_breeding_season: str
    nutrition_recommendations: List[str]
    management_suggestions: List[str]
    expected_calving_interval: Optional[str] = None


class CattleHealthAnalyzer:
    """
    Advanced health analysis system for cattle using computer vision
    """
    
    def __init__(self):
        self.breed_characteristics = self.load_breed_characteristics()
        self.nutrition_database = self.load_nutrition_database()
        self.seasonal_recommendations = self.load_seasonal_recommendations()
    
    def load_breed_characteristics(self) -> Dict:
        """Load breed-specific characteristics and standards"""
        return {
            'Gir': {
                'mature_weight_kg': {'male': (400, 500), 'female': (300, 400)},
                'height_cm': {'male': (130, 140), 'female': (120, 130)},
                'characteristics': ['distinctive hump', 'curved horns', 'heat tolerant'],
                'optimal_bcs': (5, 7),
                'breeding_season': 'year-round',
                'first_calving_age_months': 36,
                'calving_interval_months': 13
            },
            'Sahiwal': {
                'mature_weight_kg': {'male': (450, 550), 'female': (350, 450)},
                'height_cm': {'male': (132, 142), 'female': (122, 132)},
                'characteristics': ['reddish-brown color', 'loose skin', 'good milker'],
                'optimal_bcs': (5, 7),
                'breeding_season': 'year-round',
                'first_calving_age_months': 34,
                'calving_interval_months': 12
            },
            'Holstein_Friesian': {
                'mature_weight_kg': {'male': (700, 900), 'female': (550, 700)},
                'height_cm': {'male': (150, 160), 'female': (140, 150)},
                'characteristics': ['black and white patches', 'large frame', 'high milk yield'],
                'optimal_bcs': (6, 8),
                'breeding_season': 'year-round',
                'first_calving_age_months': 24,
                'calving_interval_months': 12
            },
            'Jersey': {
                'mature_weight_kg': {'male': (500, 650), 'female': (350, 450)},
                'height_cm': {'male': (125, 135), 'female': (115, 125)},
                'characteristics': ['fawn color', 'compact size', 'high butterfat'],
                'optimal_bcs': (5, 7),
                'breeding_season': 'year-round',
                'first_calving_age_months': 22,
                'calving_interval_months': 11
            },
            'Red_Sindhi': {
                'mature_weight_kg': {'male': (400, 500), 'female': (300, 400)},
                'height_cm': {'male': (130, 140), 'female': (120, 130)},
                'characteristics': ['red color', 'heat tolerant', 'dual purpose'],
                'optimal_bcs': (5, 7),
                'breeding_season': 'year-round',
                'first_calving_age_months': 36,
                'calving_interval_months': 13
            }
        }
    
    def load_nutrition_database(self) -> Dict:
        """Load nutrition requirements by breed and condition"""
        return {
            'dry_matter_intake_percent': {
                'lactating': 3.5,
                'pregnant': 2.8,
                'dry': 2.2,
                'growing': 3.0
            },
            'protein_requirements_percent': {
                'lactating': {'high_yield': 16, 'medium_yield': 14, 'low_yield': 12},
                'pregnant': 11,
                'dry': 9,
                'growing': 14
            },
            'energy_requirements_mcal_per_day': {
                'maintenance': {'small': 12, 'medium': 15, 'large': 18},
                'lactation_per_kg_milk': 0.74,
                'pregnancy_last_trimester': 3.2,
                'growth_per_kg_gain': 4.4
            },
            'mineral_requirements': {
                'calcium_percent': 0.6,
                'phosphorus_percent': 0.4,
                'salt_percent': 0.5,
                'trace_minerals': ['copper', 'zinc', 'selenium', 'cobalt']
            }
        }
    
    def load_seasonal_recommendations(self) -> Dict:
        """Load seasonal breeding and management recommendations"""
        return {
            'monsoon': {
                'breeding': 'Ideal for heat-sensitive breeds',
                'nutrition': 'Focus on green fodder, watch for parasites',
                'management': 'Ensure proper drainage, foot care'
            },
            'winter': {
                'breeding': 'Good for all breeds',
                'nutrition': 'Increase energy supplements',
                'management': 'Provide shelter, increase feeding frequency'
            },
            'summer': {
                'breeding': 'Avoid peak heat months for temperate breeds',
                'nutrition': 'Provide adequate water, electrolytes',
                'management': 'Shade, ventilation, early morning/evening feeding'
            }
        }
    
    def analyze_body_condition(self, image: np.ndarray, breed: str, gender: str = 'female') -> HealthAssessment:
        """
        Comprehensive body condition analysis using computer vision
        """
        # Extract morphological features
        body_features = self.extract_body_features(image)
        
        # Estimate body condition score
        bcs = self.estimate_body_condition_score(body_features, breed)
        
        # Categorize condition
        condition_category = self.categorize_body_condition(bcs)
        
        # Assess overall health status
        health_status = self.assess_health_status(bcs, breed, body_features)
        
        # Estimate weight if possible
        weight_estimate = self.estimate_weight(body_features, breed, gender)
        
        return HealthAssessment(
            body_condition_score=bcs,
            condition_category=condition_category,
            health_status=health_status,
            weight_estimate_kg=weight_estimate
        )
    
    def extract_body_features(self, image: np.ndarray) -> Dict:
        """
        Extract morphological features for body condition assessment
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection for body outline
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contours found'}
        
        # Get the largest contour (assumed to be the cattle body)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate basic geometric features
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Fit bounding rectangle and ellipse
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull and solidity
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Fit ellipse for body shape analysis
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            ellipse_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        else:
            ellipse_ratio = 1.0
        
        # Calculate moments for shape analysis
        moments = cv2.moments(main_contour)
        
        # Compactness (circularity)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Regional analysis for fat deposits
        body_regions = self.analyze_body_regions(gray, main_contour)
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'ellipse_ratio': ellipse_ratio,
            'compactness': compactness,
            'bounding_box': (x, y, w, h),
            'body_regions': body_regions,
            'moments': moments
        }
    
    def analyze_body_regions(self, gray_image: np.ndarray, contour: np.ndarray) -> Dict:
        """
        Analyze specific body regions for condition assessment
        """
        # Create mask from contour
        mask = np.zeros(gray_image.shape, np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Define regions of interest
        regions = {
            'spine_area': gray_image[y:y+h//3, x+w//4:x+3*w//4],
            'rib_area': gray_image[y+h//4:y+3*h//4, x:x+w//2],
            'hip_area': gray_image[y+2*h//3:y+h, x+w//4:x+3*w//4],
            'udder_area': gray_image[y+3*h//4:y+h, x+w//3:x+2*w//3]
        }
        
        region_analysis = {}
        for region_name, region_img in regions.items():
            if region_img.size > 0:
                # Calculate texture features
                mean_intensity = np.mean(region_img)
                std_intensity = np.std(region_img)
                
                # Calculate local binary pattern or other texture features
                region_analysis[region_name] = {
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'uniformity': std_intensity / mean_intensity if mean_intensity > 0 else 0
                }
        
        return region_analysis
    
    def estimate_body_condition_score(self, features: Dict, breed: str) -> float:
        """
        Estimate body condition score based on extracted features
        """
        if 'error' in features:
            return 5.0  # Default moderate score
        
        # Feature-based BCS estimation (simplified approach)
        # In a real implementation, this would use a trained model
        
        solidity = features.get('solidity', 0.5)
        ellipse_ratio = features.get('ellipse_ratio', 1.5)
        compactness = features.get('compactness', 0.5)
        
        # Simple scoring algorithm (would be replaced by ML model)
        base_score = 5.0
        
        # Adjust based on body shape
        if solidity > 0.85:  # More filled out body
            base_score += 1.0
        elif solidity < 0.75:  # Thinner body
            base_score -= 1.0
        
        # Adjust based on body proportions
        if ellipse_ratio > 2.0:  # Very elongated
            base_score -= 0.5
        elif ellipse_ratio < 1.5:  # More rounded
            base_score += 0.5
        
        # Breed-specific adjustments
        breed_chars = self.breed_characteristics.get(breed, {})
        optimal_bcs = breed_chars.get('optimal_bcs', (5, 7))
        
        # Ensure score is within valid range
        bcs = max(1.0, min(9.0, base_score))
        
        return round(bcs, 1)
    
    def categorize_body_condition(self, bcs: float) -> str:
        """Categorize body condition score"""
        if bcs <= 2:
            return "Emaciated"
        elif bcs <= 3:
            return "Thin"
        elif bcs <= 4:
            return "Moderately Thin"
        elif bcs <= 6:
            return "Moderate"
        elif bcs <= 7:
            return "Moderately Fleshy"
        elif bcs <= 8:
            return "Fleshy"
        else:
            return "Obese"
    
    def assess_health_status(self, bcs: float, breed: str, features: Dict) -> str:
        """Assess overall health status"""
        breed_chars = self.breed_characteristics.get(breed, {})
        optimal_bcs = breed_chars.get('optimal_bcs', (5, 7))
        
        if optimal_bcs[0] <= bcs <= optimal_bcs[1]:
            return "Optimal"
        elif bcs < optimal_bcs[0] - 1:
            return "Undernourished"
        elif bcs > optimal_bcs[1] + 1:
            return "Overfed"
        else:
            return "Acceptable"
    
    def estimate_weight(self, features: Dict, breed: str, gender: str = 'female') -> Optional[float]:
        """Estimate weight based on body measurements"""
        if 'error' in features:
            return None
        
        breed_chars = self.breed_characteristics.get(breed, {})
        weight_range = breed_chars.get('mature_weight_kg', {}).get(gender, (300, 400))
        
        # Simple weight estimation based on body area and breed standards
        # In practice, this would use more sophisticated measurements
        area = features.get('area', 0)
        solidity = features.get('solidity', 0.5)
        
        # Normalize area (this is a simplified approach)
        if area > 0:
            # Estimate weight as a function of body area and breed characteristics
            base_weight = (weight_range[0] + weight_range[1]) / 2
            area_factor = min(1.5, max(0.5, area / 50000))  # Normalize area
            solidity_factor = max(0.8, min(1.2, solidity))
            
            estimated_weight = base_weight * area_factor * solidity_factor
            return round(estimated_weight, 1)
        
        return None
    
    def generate_health_recommendations(self, assessment: HealthAssessment, breed: str) -> List[str]:
        """Generate health and nutrition recommendations"""
        recommendations = []
        
        bcs = assessment.body_condition_score
        condition = assessment.condition_category
        
        # BCS-based recommendations
        if bcs < 4:
            recommendations.extend([
                "Increase energy intake with high-quality concentrates",
                "Provide additional protein supplements",
                "Check for internal parasites and treat if necessary",
                "Ensure adequate fresh water availability",
                "Consider vitamin and mineral supplementation"
            ])
        elif bcs > 7:
            recommendations.extend([
                "Reduce energy-dense feeds",
                "Increase exercise and grazing time",
                "Monitor for metabolic disorders",
                "Adjust feeding frequency and portion sizes"
            ])
        else:
            recommendations.append("Maintain current feeding regimen")
        
        # Breed-specific recommendations
        breed_chars = self.breed_characteristics.get(breed, {})
        
        if 'heat tolerant' in breed_chars.get('characteristics', []):
            recommendations.append("Suitable for hot climate management")
        else:
            recommendations.append("Provide adequate cooling during hot weather")
        
        if 'good milker' in breed_chars.get('characteristics', []):
            recommendations.append("Monitor for udder health and mastitis prevention")
        
        return recommendations


class BreedingAdvisor:
    """
    Breeding recommendations and management advisor
    """
    
    def __init__(self, health_analyzer: CattleHealthAnalyzer):
        self.health_analyzer = health_analyzer
        self.breeding_calendar = self.load_breeding_calendar()
    
    def load_breeding_calendar(self) -> Dict:
        """Load seasonal breeding calendar"""
        return {
            'optimal_breeding_months': {
                'tropical_breeds': [1, 2, 3, 10, 11, 12],  # Cooler months
                'temperate_breeds': [4, 5, 6, 9, 10, 11]   # Moderate temperature months
            },
            'calving_seasons': {
                'monsoon_calving': [6, 7, 8, 9],   # Good for feed availability
                'winter_calving': [11, 12, 1, 2],  # Easier management
                'summer_calving': [3, 4, 5]        # Challenging, avoid if possible
            }
        }
    
    def generate_breeding_recommendations(self, breed: str, gender: str, age_months: int, 
                                        health_assessment: HealthAssessment, 
                                        current_month: int = None) -> BreedingRecommendation:
        """
        Generate comprehensive breeding recommendations
        """
        if current_month is None:
            current_month = datetime.now().month
        
        breed_chars = self.health_analyzer.breed_characteristics.get(breed, {})
        
        # Determine breeding readiness
        breeding_readiness = self.assess_breeding_readiness(
            gender, age_months, health_assessment, breed_chars
        )
        
        # Determine optimal breeding season
        optimal_season = self.determine_optimal_breeding_season(breed, current_month)
        
        # Generate nutrition recommendations
        nutrition_recs = self.generate_breeding_nutrition_recommendations(
            breed, gender, health_assessment
        )
        
        # Generate management suggestions
        management_suggestions = self.generate_management_suggestions(
            breed, gender, health_assessment, current_month
        )
        
        # Calculate expected calving interval
        calving_interval = breed_chars.get('calving_interval_months', 12)
        expected_calving = f"{calving_interval} months" if breeding_readiness == "Ready" else None
        
        return BreedingRecommendation(
            breeding_readiness=breeding_readiness,
            optimal_breeding_season=optimal_season,
            nutrition_recommendations=nutrition_recs,
            management_suggestions=management_suggestions,
            expected_calving_interval=expected_calving
        )
    
    def assess_breeding_readiness(self, gender: str, age_months: int, 
                                health_assessment: HealthAssessment, 
                                breed_chars: Dict) -> str:
        """Assess if animal is ready for breeding"""
        
        if gender == 'male':
            min_age = 18  # Bulls generally mature earlier
            if age_months < min_age:
                return f"Too young - wait {min_age - age_months} months"
        else:  # female
            min_age = breed_chars.get('first_calving_age_months', 24) - 9  # Subtract gestation
            if age_months < min_age:
                return f"Too young - wait {min_age - age_months} months"
        
        # Check body condition
        bcs = health_assessment.body_condition_score
        optimal_bcs = breed_chars.get('optimal_bcs', (5, 7))
        
        if bcs < optimal_bcs[0]:
            return "Improve body condition before breeding"
        elif bcs > optimal_bcs[1] + 1:
            return "Reduce body condition before breeding"
        elif health_assessment.health_status != "Optimal":
            return "Address health issues before breeding"
        else:
            return "Ready for breeding"
    
    def determine_optimal_breeding_season(self, breed: str, current_month: int) -> str:
        """Determine optimal breeding season"""
        
        # Classify breed type
        breed_chars = self.health_analyzer.breed_characteristics.get(breed, {})
        characteristics = breed_chars.get('characteristics', [])
        
        if 'heat tolerant' in characteristics:
            breed_type = 'tropical_breeds'
        else:
            breed_type = 'temperate_breeds'
        
        optimal_months = self.breeding_calendar['optimal_breeding_months'][breed_type]
        
        if current_month in optimal_months:
            return "Current month is optimal for breeding"
        else:
            # Find next optimal month
            next_optimal = None
            for month in optimal_months:
                if month > current_month:
                    next_optimal = month
                    break
            
            if next_optimal is None:
                next_optimal = optimal_months[0] + 12  # Next year
            
            months_to_wait = next_optimal - current_month
            return f"Wait {months_to_wait} months for optimal breeding season"
    
    def generate_breeding_nutrition_recommendations(self, breed: str, gender: str, 
                                                  health_assessment: HealthAssessment) -> List[str]:
        """Generate nutrition recommendations for breeding animals"""
        
        recommendations = []
        bcs = health_assessment.body_condition_score
        
        # Pre-breeding nutrition
        if gender == 'female':
            recommendations.extend([
                "Increase protein intake 2-3 months before breeding",
                "Ensure adequate vitamin A and E supplementation",
                "Provide minerals especially calcium and phosphorus",
                "Maintain consistent body weight during breeding period"
            ])
            
            if bcs < 5:
                recommendations.append("Increase energy intake to improve body condition")
            elif bcs > 7:
                recommendations.append("Slightly reduce energy intake to prevent over-conditioning")
        
        else:  # male
            recommendations.extend([
                "High-quality protein for semen production",
                "Zinc and selenium supplementation for fertility",
                "Adequate vitamin E for reproductive health",
                "Maintain optimal body condition for breeding soundness"
            ])
        
        # Breed-specific nutrition
        breed_chars = self.health_analyzer.breed_characteristics.get(breed, {})
        if 'high milk yield' in breed_chars.get('characteristics', []):
            recommendations.append("Plan for high-energy lactation diet post-calving")
        
        return recommendations
    
    def generate_management_suggestions(self, breed: str, gender: str, 
                                      health_assessment: HealthAssessment, 
                                      current_month: int) -> List[str]:
        """Generate management suggestions for breeding"""
        
        suggestions = []
        
        # General breeding management
        if gender == 'female':
            suggestions.extend([
                "Monitor estrus cycles regularly",
                "Maintain breeding records and calendar",
                "Ensure proper vaccination schedule",
                "Provide comfortable, clean breeding environment"
            ])
        else:  # male
            suggestions.extend([
                "Regular breeding soundness examination",
                "Maintain proper hoof care",
                "Provide adequate exercise and conditioning",
                "Monitor for any breeding injuries"
            ])
        
        # Seasonal management
        season_recs = self.health_analyzer.seasonal_recommendations
        
        if 4 <= current_month <= 6:  # Summer
            suggestions.extend([
                "Provide adequate shade and cooling",
                "Breed during cooler parts of the day",
                "Ensure constant access to fresh water"
            ])
        elif 11 <= current_month <= 2:  # Winter  
            suggestions.extend([
                "Provide shelter from cold winds",
                "Increase feeding frequency",
                "Monitor for seasonal breeding advantages"
            ])
        else:  # Monsoon
            suggestions.extend([
                "Ensure proper drainage in breeding areas",
                "Monitor for increased parasite load",
                "Take advantage of green fodder availability"
            ])
        
        # Health-based suggestions
        if health_assessment.health_status != "Optimal":
            suggestions.append("Consult veterinarian before breeding")
        
        return suggestions


class ComprehensiveHealthSystem:
    """
    Integrated system combining health analysis and breeding recommendations
    """
    
    def __init__(self):
        self.health_analyzer = CattleHealthAnalyzer()
        self.breeding_advisor = BreedingAdvisor(self.health_analyzer)
    
    def comprehensive_assessment(self, image: np.ndarray, breed: str, 
                                gender: str = 'female', age_months: int = 24) -> Dict:
        """
        Perform comprehensive health and breeding assessment
        """
        print(f"🔍 Analyzing {breed} cattle health and breeding status...")
        
        # Health assessment
        health_assessment = self.health_analyzer.analyze_body_condition(image, breed, gender)
        
        # Health recommendations
        health_recommendations = self.health_analyzer.generate_health_recommendations(
            health_assessment, breed
        )
        
        # Breeding recommendations
        breeding_recommendations = self.breeding_advisor.generate_breeding_recommendations(
            breed, gender, age_months, health_assessment
        )
        
        # Compile comprehensive report
        report = {
            'assessment_date': datetime.now().isoformat(),
            'animal_info': {
                'breed': breed,
                'gender': gender,
                'estimated_age_months': age_months
            },
            'health_assessment': {
                'body_condition_score': health_assessment.body_condition_score,
                'condition_category': health_assessment.condition_category,
                'health_status': health_assessment.health_status,
                'weight_estimate_kg': health_assessment.weight_estimate_kg
            },
            'health_recommendations': health_recommendations,
            'breeding_assessment': {
                'breeding_readiness': breeding_recommendations.breeding_readiness,
                'optimal_breeding_season': breeding_recommendations.optimal_breeding_season,
                'expected_calving_interval': breeding_recommendations.expected_calving_interval
            },
            'nutrition_recommendations': breeding_recommendations.nutrition_recommendations,
            'management_suggestions': breeding_recommendations.management_suggestions
        }
        
        return report
    
    def save_assessment_report(self, report: Dict, output_path: str):
        """Save assessment report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📁 Assessment report saved: {output_path}")
    
    def generate_summary_report(self, reports: List[Dict]) -> Dict:
        """Generate summary report from multiple assessments"""
        
        if not reports:
            return {'error': 'No reports to summarize'}
        
        # Aggregate statistics
        total_animals = len(reports)
        breed_distribution = {}
        bcs_scores = []
        health_statuses = {}
        breeding_ready_count = 0
        
        for report in reports:
            # Breed distribution
            breed = report['animal_info']['breed']
            breed_distribution[breed] = breed_distribution.get(breed, 0) + 1
            
            # BCS scores
            bcs = report['health_assessment']['body_condition_score']
            bcs_scores.append(bcs)
            
            # Health status
            health_status = report['health_assessment']['health_status']
            health_statuses[health_status] = health_statuses.get(health_status, 0) + 1
            
            # Breeding readiness
            if report['breeding_assessment']['breeding_readiness'] == 'Ready for breeding':
                breeding_ready_count += 1
        
        summary = {
            'summary_date': datetime.now().isoformat(),
            'total_animals_assessed': total_animals,
            'breed_distribution': breed_distribution,
            'health_statistics': {
                'average_bcs': np.mean(bcs_scores),
                'bcs_std': np.std(bcs_scores),
                'health_status_distribution': health_statuses
            },
            'breeding_statistics': {
                'animals_ready_for_breeding': breeding_ready_count,
                'breeding_readiness_percentage': (breeding_ready_count / total_animals) * 100
            },
            'recommendations': {
                'priority_actions': self.identify_priority_actions(reports),
                'herd_management_suggestions': self.generate_herd_suggestions(reports)
            }
        }
        
        return summary
    
    def identify_priority_actions(self, reports: List[Dict]) -> List[str]:
        """Identify priority actions based on assessment results"""
        
        actions = []
        undernourished_count = 0
        overweight_count = 0
        health_issues_count = 0
        
        for report in reports:
            health_status = report['health_assessment']['health_status']
            bcs = report['health_assessment']['body_condition_score']
            
            if health_status == 'Undernourished':
                undernourished_count += 1
            elif health_status == 'Overfed':
                overweight_count += 1
            elif health_status != 'Optimal':
                health_issues_count += 1
        
        total_animals = len(reports)
        
        if undernourished_count > total_animals * 0.2:
            actions.append(f"Address malnutrition in {undernourished_count} animals")
        
        if overweight_count > total_animals * 0.1:
            actions.append(f"Implement weight management for {overweight_count} animals")
        
        if health_issues_count > total_animals * 0.3:
            actions.append("Schedule veterinary examination for herd health assessment")
        
        return actions
    
    def generate_herd_suggestions(self, reports: List[Dict]) -> List[str]:
        """Generate herd-level management suggestions"""
        
        suggestions = [
            "Implement regular body condition scoring program",
            "Maintain detailed breeding and health records",
            "Develop seasonal feeding management plan",
            "Establish regular veterinary health monitoring"
        ]
        
        # Add breed-specific suggestions
        breeds = set(report['animal_info']['breed'] for report in reports)
        
        if len(breeds) > 1:
            suggestions.append("Consider breed-specific management protocols for mixed herds")
        
        return suggestions


# Example usage and testing
if __name__ == "__main__":
    print("🏥 Testing Comprehensive Health Assessment System...")
    
    # Initialize system
    health_system = ComprehensiveHealthSystem()
    
    # Test with mock image data
    mock_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Test assessment for different breeds
    test_breeds = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal']
    
    for breed in test_breeds:
        print(f"📊 Testing assessment for {breed}...")
        
        # Perform comprehensive assessment
        report = health_system.comprehensive_assessment(
            mock_image, breed, gender='female', age_months=30
        )
        
        print(f"  BCS: {report['health_assessment']['body_condition_score']}")
        print(f"  Health Status: {report['health_assessment']['health_status']}")
        print(f"  Breeding Readiness: {report['breeding_assessment']['breeding_readiness']}")
    
    print("\n✅ Health assessment system ready!")
    print("🎯 Features available:")
    print("  ✅ Body condition scoring using computer vision")
    print("  ✅ Breed-specific health recommendations")
    print("  ✅ Comprehensive breeding advisory")
    print("  ✅ Nutrition and management suggestions")
    print("  ✅ Herd-level analysis and reporting")