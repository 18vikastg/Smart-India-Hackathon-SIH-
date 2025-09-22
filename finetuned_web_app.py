#!/usr/bin/env python3
"""
Advanced Cattle Breed Recognition Web App
Using the fine-tuned ResNet50 model trained on your dataset
"""

from flask import Flask, render_template, request, flash, jsonify, session, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-for-cattle-recognition'  # Required for sessions

class FineTunedCattleClassifier:
    """Fine-tuned cattle breed classifier"""
    
    def __init__(self, model_path='best_cattle_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.class_names = checkpoint['class_names']
            
            # Create model architecture
            self.model = resnet50()
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.model_accuracy = checkpoint.get('val_acc', 0)
            self.is_loaded = True
            
            print(f"✅ Fine-tuned model loaded! Accuracy: {self.model_accuracy:.1f}%")
            
        except Exception as e:
            print(f"❌ Could not load fine-tuned model: {e}")
            self.is_loaded = False
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_breed(self, image):
        """Predict cattle breed using fine-tuned model"""
        if not self.is_loaded:
            return {
                'error': 'Fine-tuned model not available',
                'suggestions': ['Run cattle_finetuning.py first to train the model']
            }
        
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get all predictions
            results = []
            for i, class_name in enumerate(self.class_names):
                confidence = probabilities[i].item() * 100
                results.append({
                    'breed': class_name,
                    'confidence': confidence,
                    'percentage': f"{confidence:.1f}%"
                })
            
            # Sort by confidence
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            
            return {
                'success': True,
                'predictions': results,
                'top_breed': results[0]['breed'],
                'top_confidence': results[0]['confidence'],
                'model_accuracy': f"{self.model_accuracy:.1f}%",
                'model_type': 'Fine-tuned ResNet50',
                'training_data': '1,208 European cattle images'
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'suggestions': ['Check if image is valid', 'Try a different image']
            }

# Initialize the classifier
classifier = FineTunedCattleClassifier()

def get_breed_info(breed_name):
    """Get information about the predicted breed"""
    breed_info = {
        'Ayrshire cattle': {
            'origin': 'Scotland',
            'characteristics': 'Red and white coloring, hardy breed, excellent milk production',
            'milk_production': 'High quality milk with good butterfat content',
            'temperament': 'Docile and easy to handle'
        },
        'Brown Swiss cattle': {
            'origin': 'Switzerland',
            'characteristics': 'Light to dark brown color, large size, dual-purpose breed',
            'milk_production': 'Excellent milk production with high protein content',
            'temperament': 'Calm and gentle disposition'
        },
        'Holstein Friesian cattle': {
            'origin': 'Netherlands/Germany',
            'characteristics': 'Black and white markings, large frame, world\'s highest milk producer',
            'milk_production': 'Exceptional milk yield, industry standard for dairy',
            'temperament': 'Generally docile but can be more active'
        },
        'Jersey cattle': {
            'origin': 'Jersey Island (Channel Islands)',
            'characteristics': 'Light brown to fawn color, smaller size, high butterfat milk',
            'milk_production': 'Rich, creamy milk with high butterfat and protein',
            'temperament': 'Gentle and friendly, easy to manage'
        },
        'Red Dane cattle': {
            'origin': 'Denmark',
            'characteristics': 'Red color, good size, dual-purpose for milk and meat',
            'milk_production': 'Good milk production with decent butterfat content',
            'temperament': 'Calm and manageable'
        }
    }
    
    return breed_info.get(breed_name, {
        'origin': 'Information not available',
        'characteristics': 'Breed-specific information being updated',
        'milk_production': 'Milk production data not available',
        'temperament': 'Temperament information not available'
    })

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         model_loaded=classifier.is_loaded,
                         model_accuracy=getattr(classifier, 'model_accuracy', 0))

@app.route('/demo')
def demo():
    """Demo page with sample images"""
    return render_template('demo.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        # Load and process image
        image = Image.open(filepath).convert('RGB')
        
        # Analyze with fine-tuned model
        results = classifier.predict_breed(image)
        
        if 'error' in results:
            return jsonify(results)
        
        # Add breed information
        top_breed = results['top_breed']
        breed_info = get_breed_info(top_breed)
        results['breed_info'] = breed_info
        
        # Store filename for image display (instead of base64 data)
        results['filename'] = filename
        results['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store results in session for display (without large image data)
        session['results'] = results
        
        # Redirect to results page instead of returning JSON
        return redirect(url_for('results'))
        
    except Exception as e:
        return jsonify({
            'error': f'Upload failed: {str(e)}',
            'suggestions': ['Check if file is a valid image', 'Try a smaller file size']
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results')
def results():
    """Results page"""
    # Get results from session
    results_data = session.get('results', None)
    if not results_data:
        flash('No results found. Please upload an image first.', 'error')
        return redirect(url_for('index'))
    
    # Transform the data to match the template expectations
    result = {
        'filename': results_data.get('filename', 'Unknown'),
        'upload_time': results_data.get('upload_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        'top_breed': results_data.get('top_breed', 'Unknown'),
        'top_confidence': results_data.get('top_confidence', 0),
        'image_url': f"/uploads/{results_data.get('filename', '')}" if results_data.get('filename') else '',
        'predictions': results_data.get('predictions', []),
        'breed_info': results_data.get('breed_info', {}),
        'model_accuracy': results_data.get('model_accuracy', '0%'),
        'model_type': results_data.get('model_type', 'Unknown'),
        'training_data': results_data.get('training_data', 'Unknown')
    }
    
    return render_template('results_simple.html', result=result)

if __name__ == '__main__':
    print("🐄 FINE-TUNED CATTLE BREED RECOGNITION WEB APP")
    print("=" * 55)
    print(f"🧠 Model Status: {'✅ Loaded' if classifier.is_loaded else '❌ Not Available'}")
    if classifier.is_loaded:
        print(f"🎯 Model Accuracy: {classifier.model_accuracy:.1f}%")
        print(f"🏷️  Classes: {len(classifier.class_names)} breeds")
    else:
        print("💡 Run 'python cattle_finetuning.py' first to train the model")
    print(f"📱 Starting server...")
    print(f"🌐 Visit: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)