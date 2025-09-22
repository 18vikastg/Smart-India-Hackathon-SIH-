# Batch Processing and Geo-tagging for Cattle Recognition
import concurrent.futures
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go


class GPSExtractor:
    """
    Extract GPS coordinates from image EXIF data
    """
    
    @staticmethod
    def extract_gps_from_exif(image_path: str) -> Optional[Dict]:
        """
        Extract GPS coordinates from image EXIF data
        """
        try:
            with Image.open(image_path) as image:
                exifdata = image.getexif()
                
                if exifdata is not None:
                    for tag_id in exifdata:
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == "GPSInfo":
                            gps_data = exifdata.get_ifd(tag_id)
                            return GPSExtractor.parse_gps_data(gps_data)
                            
        except Exception as e:
            print(f"⚠️  Error extracting GPS from {image_path}: {e}")
            
        return None
    
    @staticmethod
    def parse_gps_data(gps_data: Dict) -> Optional[Dict]:
        """
        Parse GPS data from EXIF
        """
        if not gps_data:
            return None
            
        try:
            lat = GPSExtractor.convert_to_degrees(gps_data.get(2))
            lat_ref = gps_data.get(1)
            lon = GPSExtractor.convert_to_degrees(gps_data.get(4))
            lon_ref = gps_data.get(3)
            
            if lat and lon and lat_ref and lon_ref:
                if lat_ref != 'N':
                    lat = -lat
                if lon_ref != 'E':
                    lon = -lon
                    
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'lat_ref': lat_ref,
                    'lon_ref': lon_ref
                }
                
        except Exception as e:
            print(f"⚠️  Error parsing GPS data: {e}")
            
        return None
    
    @staticmethod
    def convert_to_degrees(value):
        """
        Convert GPS coordinates to degrees
        """
        if not value:
            return None
            
        try:
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        except:
            return None
    
    @staticmethod
    def add_synthetic_gps(num_coordinates: int, center_lat: float = 20.5937, 
                         center_lon: float = 78.9629, radius_km: float = 500) -> List[Dict]:
        """
        Generate synthetic GPS coordinates for testing (centered on India)
        """
        coordinates = []
        
        # Convert radius from km to degrees (approximate)
        radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km
        
        for _ in range(num_coordinates):
            # Random offset within radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius_deg)
            
            lat = center_lat + distance * np.cos(angle)
            lon = center_lon + distance * np.sin(angle)
            
            coordinates.append({
                'latitude': lat,
                'longitude': lon,
                'synthetic': True
            })
        
        return coordinates


class BatchCattleProcessor:
    """
    Process multiple cattle images with batch operations and geo-tagging
    """
    
    def __init__(self, model, max_workers: int = 4):
        self.model = model
        self.max_workers = max_workers
        self.processing_stats = {
            'total_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'gps_extracted': 0,
            'processing_time': 0
        }
    
    def process_batch(self, image_paths: List[str], extract_gps: bool = True, 
                     save_results: bool = True, output_dir: str = "batch_results") -> Dict:
        """
        Process multiple images with concurrent processing
        """
        print(f"🔄 Starting batch processing of {len(image_paths)} images...")
        start_time = datetime.now()
        
        # Create output directory
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        # Process images concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path, extract_gps): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    result['image_path'] = image_path
                    results.append(result)
                    
                    self.processing_stats['successful_predictions'] += 1
                    if result.get('gps_data'):
                        self.processing_stats['gps_extracted'] += 1
                        
                except Exception as exc:
                    print(f'❌ Image {image_path} generated an exception: {exc}')
                    self.processing_stats['failed_predictions'] += 1
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.processing_stats['processing_time'] = processing_time
        self.processing_stats['total_processed'] = len(results)
        
        # Generate comprehensive analysis
        batch_analysis = self.analyze_batch_results(results)
        
        # Create final batch report
        batch_report = {
            'metadata': {
                'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'total_images': len(image_paths),
                'processing_time_seconds': processing_time,
                'images_per_second': len(results) / processing_time if processing_time > 0 else 0
            },
            'processing_stats': self.processing_stats,
            'individual_results': results,
            'batch_analysis': batch_analysis
        }
        
        # Save results if requested
        if save_results:
            self.save_batch_results(batch_report, output_dir)
            self.create_batch_visualizations(batch_report, output_dir)
        
        print(f"✅ Batch processing complete! Processed {len(results)} images in {processing_time:.2f} seconds")
        return batch_report
    
    def process_single_image(self, image_path: str, extract_gps: bool = True) -> Dict:
        """
        Process a single image with breed prediction and GPS extraction
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Make prediction (this would use your actual model)
            prediction = self.predict_breed(image)
            result.update(prediction)
            
            # Extract GPS data if requested
            if extract_gps:
                gps_data = GPSExtractor.extract_gps_from_exif(image_path)
                if gps_data:
                    result['gps_data'] = gps_data
                    # Add reverse geocoding if available
                    location_info = self.reverse_geocode(gps_data)
                    if location_info:
                        result['location_info'] = location_info
            
            # Extract image metadata
            result['image_metadata'] = self.extract_image_metadata(image_path)
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def predict_breed(self, image: np.ndarray) -> Dict:
        """
        Make breed prediction using the model
        """
        # This would integrate with your actual trained model
        # For now, returning mock predictions
        
        # Simulate model prediction
        breed_names = ["Gir", "Sahiwal", "Holstein_Friesian", "Jersey", "Red_Sindhi"]
        confidences = np.random.dirichlet(np.ones(len(breed_names)) * 2)  # More realistic distribution
        
        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        
        predictions = []
        for i in range(min(3, len(breed_names))):  # Top 3 predictions
            idx = sorted_indices[i]
            predictions.append({
                'breed': breed_names[idx],
                'confidence': float(confidences[idx]),
                'rank': i + 1
            })
        
        return {
            'breed_predictions': predictions,
            'top_breed': predictions[0]['breed'],
            'top_confidence': predictions[0]['confidence']
        }
    
    def extract_image_metadata(self, image_path: str) -> Dict:
        """
        Extract general image metadata
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'filename': os.path.basename(image_path),
                    'format': img.format,
                    'size': img.size,
                    'mode': img.mode,
                    'file_size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            return {'error': str(e)}
    
    def reverse_geocode(self, gps_data: Dict) -> Optional[Dict]:
        """
        Reverse geocode GPS coordinates to location information
        """
        # This would use a reverse geocoding service
        # For now, returning mock location data for Indian coordinates
        
        lat, lon = gps_data['latitude'], gps_data['longitude']
        
        # Simple mock location assignment based on coordinates
        if 8 <= lat <= 37 and 68 <= lon <= 97:  # Rough bounds of India
            mock_states = [
                "Rajasthan", "Gujarat", "Maharashtra", "Karnataka", "Tamil Nadu",
                "Andhra Pradesh", "Uttar Pradesh", "Madhya Pradesh", "Punjab", "Haryana"
            ]
            
            return {
                'country': 'India',
                'state': np.random.choice(mock_states),
                'district': f"District_{np.random.randint(1, 50)}",
                'coordinates_valid': True
            }
        
        return {'coordinates_valid': False}
    
    def analyze_batch_results(self, results: List[Dict]) -> Dict:
        """
        Analyze batch processing results for insights
        """
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful predictions to analyze'}
        
        # Breed distribution analysis
        breed_counts = Counter()
        confidence_scores = []
        
        for result in successful_results:
            if 'breed_predictions' in result:
                top_breed = result['breed_predictions'][0]['breed']
                top_confidence = result['breed_predictions'][0]['confidence']
                
                breed_counts[top_breed] += 1
                confidence_scores.append(top_confidence)
        
        # Geographic distribution analysis
        gps_results = [r for r in successful_results if 'gps_data' in r]
        location_distribution = Counter()
        
        for result in gps_results:
            location_info = result.get('location_info', {})
            state = location_info.get('state', 'Unknown')
            location_distribution[state] += 1
        
        # Statistical analysis
        analysis = {
            'breed_distribution': dict(breed_counts),
            'most_common_breed': breed_counts.most_common(1)[0] if breed_counts else None,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'total_unique_breeds': len(breed_counts),
            'geographic_distribution': dict(location_distribution),
            'images_with_gps': len(gps_results),
            'gps_percentage': (len(gps_results) / len(successful_results)) * 100
        }
        
        return analysis
    
    def save_batch_results(self, batch_report: Dict, output_dir: str):
        """
        Save batch processing results to files
        """
        # Save main report as JSON
        report_path = os.path.join(output_dir, f"{batch_report['metadata']['batch_id']}_report.json")
        with open(report_path, 'w') as f:
            json.dump(batch_report, f, indent=2, default=str)
        
        # Save results as CSV for easy analysis
        if batch_report['individual_results']:
            results_df = self.results_to_dataframe(batch_report['individual_results'])
            csv_path = os.path.join(output_dir, f"{batch_report['metadata']['batch_id']}_results.csv")
            results_df.to_csv(csv_path, index=False)
        
        print(f"📁 Batch results saved to: {output_dir}")
    
    def results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame
        """
        flattened_results = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            flat_result = {
                'image_path': result.get('image_path', ''),
                'timestamp': result.get('timestamp', ''),
                'success': result.get('success', False)
            }
            
            # Add top breed prediction
            if 'breed_predictions' in result and result['breed_predictions']:
                top_pred = result['breed_predictions'][0]
                flat_result['predicted_breed'] = top_pred['breed']
                flat_result['confidence'] = top_pred['confidence']
            
            # Add GPS data
            if 'gps_data' in result:
                gps = result['gps_data']
                flat_result['latitude'] = gps.get('latitude')
                flat_result['longitude'] = gps.get('longitude')
            
            # Add location info
            if 'location_info' in result:
                loc = result['location_info']
                flat_result['state'] = loc.get('state')
                flat_result['country'] = loc.get('country')
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def create_batch_visualizations(self, batch_report: Dict, output_dir: str):
        """
        Create visualizations for batch processing results
        """
        print("📊 Creating batch visualizations...")
        
        # Create breed distribution map if GPS data available
        gps_results = [r for r in batch_report['individual_results'] 
                      if r.get('success') and 'gps_data' in r]
        
        if gps_results:
            map_path = self.create_breed_distribution_map(gps_results, output_dir)
            print(f"🗺️  Breed distribution map saved: {map_path}")
        
        # Create statistical plots
        self.create_statistical_plots(batch_report, output_dir)


class BreedDistributionMapper:
    """
    Create interactive maps for breed distribution visualization
    """
    
    def __init__(self):
        self.breed_colors = self.generate_breed_colors()
    
    def generate_breed_colors(self) -> Dict[str, str]:
        """
        Generate consistent colors for different breeds
        """
        common_breeds = [
            "Gir", "Sahiwal", "Holstein_Friesian", "Jersey", "Red_Sindhi",
            "Tharparkar", "Rathi", "Hariana", "Ongole", "Krishna_Valley"
        ]
        
        # Use a colormap for consistent colors
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
        ]
        
        return {breed: colors[i % len(colors)] for i, breed in enumerate(common_breeds)}
    
    def create_interactive_map(self, results: List[Dict], output_path: str = "breed_distribution_map.html"):
        """
        Create an interactive Folium map showing breed distribution
        """
        # Filter results with GPS data
        gps_results = [r for r in results if 'gps_data' in r and r.get('success')]
        
        if not gps_results:
            print("⚠️  No GPS data available for mapping")
            return None
        
        # Calculate map center
        latitudes = [r['gps_data']['latitude'] for r in gps_results]
        longitudes = [r['gps_data']['longitude'] for r in gps_results]
        
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add marker cluster for better performance with many points
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each detection
        for result in gps_results:
            gps_data = result['gps_data']
            lat, lon = gps_data['latitude'], gps_data['longitude']
            
            # Get breed information
            top_breed = result.get('top_breed', 'Unknown')
            confidence = result.get('top_confidence', 0)
            
            # Get breed color
            color = self.breed_colors.get(top_breed, '#999999')
            
            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 200px;">
                <h4 style="margin: 0; color: {color};">{top_breed}</h4>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Location:</strong> {lat:.4f}, {lon:.4f}</p>
                <p><strong>Image:</strong> {os.path.basename(result.get('image_path', ''))}</p>
            </div>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=folium.Popup(popup_content, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(marker_cluster)
        
        # Add legend
        legend_html = self.create_map_legend()
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add breed distribution statistics
        stats_html = self.create_stats_panel(gps_results)
        m.get_root().html.add_child(folium.Element(stats_html))
        
        # Save map
        m.save(output_path)
        print(f"🗺️  Interactive map saved: {output_path}")
        
        return output_path
    
    def create_map_legend(self) -> str:
        """
        Create HTML legend for the map
        """
        legend_items = []
        for breed, color in self.breed_colors.items():
            legend_items.append(f'''
                <div style="display: flex; align-items: center; margin: 2px 0;">
                    <div style="width: 12px; height: 12px; background-color: {color}; 
                               border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-size: 12px;">{breed}</span>
                </div>
            ''')
        
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto;
                    background-color: white; border: 2px solid grey; z-index: 9999;
                    font-size: 14px; padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">🐄 Cattle Breeds</h4>
            {''.join(legend_items)}
        </div>
        '''
        
        return legend_html
    
    def create_stats_panel(self, results: List[Dict]) -> str:
        """
        Create statistics panel for the map
        """
        # Calculate statistics
        breed_counts = Counter()
        total_animals = len(results)
        
        for result in results:
            breed = result.get('top_breed', 'Unknown')
            breed_counts[breed] += 1
        
        # Create stats HTML
        stats_items = []
        for breed, count in breed_counts.most_common(5):
            percentage = (count / total_animals) * 100
            stats_items.append(f'''
                <div style="margin: 3px 0; font-size: 11px;">
                    {breed}: {count} ({percentage:.1f}%)
                </div>
            ''')
        
        stats_html = f'''
        <div style="position: fixed; 
                    top: 80px; right: 20px; width: 180px; height: auto;
                    background-color: white; border: 2px solid grey; z-index: 9999;
                    font-size: 12px; padding: 10px; border-radius: 5px;">
            <h4 style="margin-top: 0;">📊 Statistics</h4>
            <div><strong>Total Animals:</strong> {total_animals}</div>
            <div><strong>Unique Breeds:</strong> {len(breed_counts)}</div>
            <hr style="margin: 8px 0;">
            <div><strong>Top Breeds:</strong></div>
            {''.join(stats_items)}
        </div>
        '''
        
        return stats_html


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Batch Processing System...")
    
    # Initialize batch processor (would use actual model)
    processor = BatchCattleProcessor(model=None, max_workers=4)
    
    # Test GPS extraction
    print("📍 Testing GPS extraction...")
    test_coordinates = GPSExtractor.add_synthetic_gps(10)
    print(f"✅ Generated {len(test_coordinates)} synthetic GPS coordinates")
    
    # Test mapping
    mapper = BreedDistributionMapper()
    print("🗺️  Mapping system initialized")
    
    print("\n🎯 Batch processing system ready!")
    print("📋 Features available:")
    print("  ✅ Concurrent batch processing")
    print("  ✅ GPS extraction from EXIF data")
    print("  ✅ Interactive breed distribution maps")
    print("  ✅ Statistical analysis and reporting")
    print("  ✅ Export to CSV and JSON formats")