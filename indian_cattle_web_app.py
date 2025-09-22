#!/usr/bin/env python3
"""
Indian Cattle Breed Recognition Web App
Using the fine-tuned ResNet50 model trained on 41 Indian bovine breeds
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
app.secret_key = 'indian-cattle-recognition-secret-key-2025'  # Required for sessions

class IndianCattleClassifier:
    """Indian cattle breed classifier using fine-tuned ResNet50"""
    
    def __init__(self, model_path='best_indian_cattle_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.is_loaded = False
        self.model_accuracy = 0.0
        
        # Data preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the fine-tuned model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("💡 Please run 'python indian_cattle_finetuning.py' first to train the model")
                return
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model info
            self.class_names = checkpoint['class_names']
            num_classes = checkpoint['num_classes']
            self.model_accuracy = checkpoint.get('accuracy', 0.0)
            
            # Build model architecture
            self.model = resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ Indian cattle model loaded! Accuracy: {self.model_accuracy:.1f}%")
            print(f"🏷️  Loaded {len(self.class_names)} Indian breeds:")
            for i, breed in enumerate(self.class_names[:10]):  # Show first 10
                print(f"   {i+1:2d}. {breed}")
            if len(self.class_names) > 10:
                print(f"   ... and {len(self.class_names) - 10} more breeds")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def predict_breed(self, image):
        """Predict cattle breed from image"""
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            
            # Transform image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
                breed_name = self.class_names[idx.item()]
                confidence = prob.item() * 100
                predictions.append({
                    'breed': breed_name,
                    'confidence': confidence,
                    'percentage': f"{confidence:.1f}%"
                })
            
            # Return results
            return {
                'success': True,
                'top_breed': predictions[0]['breed'],
                'top_confidence': predictions[0]['confidence'],
                'predictions': predictions,
                'model_accuracy': f"{self.model_accuracy:.1f}%",
                'model_type': 'Fine-tuned ResNet50',
                'training_data': f"{len(self.class_names)} Indian breeds"
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'suggestions': ['Check if the image is valid', 'Try a different image format']
            }

# Initialize the classifier
classifier = IndianCattleClassifier()

def get_indian_breed_info(breed_name):
    """Get information about Indian cattle breeds"""
    
    # Indian cattle breed information database
    breed_info = {
        'Gir': {
            'origin': 'Gujarat, India',
            'characteristics': 'Distinctive lyre-shaped horns, drooping ears, humped back',
            'milk_production': 'High milk yield, A2 milk quality',
            'temperament': 'Docile and hardy',
            'special_features': 'Heat resistant, disease resistant'
        },
        'Sahiwal': {
            'origin': 'Punjab and Haryana, India',
            'characteristics': 'Reddish-brown color, loose skin, large dewlap',
            'milk_production': 'Excellent milk producer',
            'temperament': 'Calm and gentle',
            'special_features': 'Tick resistant, heat tolerant'
        },
        'Tharparkar': {
            'origin': 'Rajasthan, India',
            'characteristics': 'White or light grey color, medium to large size',
            'milk_production': 'Good dual-purpose breed',
            'temperament': 'Hardy and adaptable',
            'special_features': 'Drought resistant'
        },
        'Red_Sindhi': {
            'origin': 'Sindh region (now Pakistan), popular in India',
            'characteristics': 'Red colored, compact body, good conformation',
            'milk_production': 'High milk fat content',
            'temperament': 'Active and alert',
            'special_features': 'Heat tolerant, good grazer'
        },
        'Hariana': {
            'origin': 'Haryana, India',
            'characteristics': 'Grey-white color, strong build, good draught animal',
            'milk_production': 'Moderate milk yield',
            'temperament': 'Strong and hardworking',
            'special_features': 'Excellent for bullock work'
        },
        'Ongole': {
            'origin': 'Andhra Pradesh, India',
            'characteristics': 'Large size, white color, prominent hump',
            'milk_production': 'Moderate milk production',
            'temperament': 'Robust and sturdy',
            'special_features': 'Heat and disease resistant'
        },
        'Kankrej': {
            'origin': 'Gujarat and Rajasthan, India',
            'characteristics': 'Silver-grey color, lyre-shaped horns',
            'milk_production': 'Good milk producer',
            'temperament': 'Active and intelligent',
            'special_features': 'Good draught animal'
        },
        'Rathi': {
            'origin': 'Rajasthan, India',
            'characteristics': 'Brown and white patches, medium size',
            'milk_production': 'Good milk yield',
            'temperament': 'Hardy and adaptable',
            'special_features': 'Desert adapted'
        },
        'Deoni': {
            'origin': 'Maharashtra and Karnataka, India',
            'characteristics': 'Spotted black and white, strong build',
            'milk_production': 'Dual-purpose breed',
            'temperament': 'Hardy and disease resistant',
            'special_features': 'Good for both milk and draught'
        },
        'Hallikar': {
            'origin': 'Karnataka, India',
            'characteristics': 'Grey color, compact body, good workers',
            'milk_production': 'Moderate milk production',
            'temperament': 'Strong and energetic',
            'special_features': 'Excellent bullock breed'
        }
    }
    
    # Default info for breeds not in database
    default_info = {
        'origin': 'India',
        'characteristics': 'Indigenous Indian cattle breed',
        'milk_production': 'Traditional dairy breed',
        'temperament': 'Hardy and well-adapted',
        'special_features': 'Heat and disease resistant'
    }
    
    return breed_info.get(breed_name, default_info)

@app.route('/')
def index():
    """Main page"""
    return render_template('indian_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'})
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        # Load and process image
        image = Image.open(filepath).convert('RGB')
        
        # Analyze with Indian cattle model
        results = classifier.predict_breed(image)
        
        if 'error' in results:
            return jsonify(results)
        
        # Add breed information
        top_breed = results['top_breed']
        breed_info = get_indian_breed_info(top_breed)
        results['breed_info'] = breed_info
        
        # Store filename for image display
        results['filename'] = filename
        results['upload_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store results in session for display (without large image data)
        session['results'] = results
        
        # Redirect to results page
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
    
    return render_template('indian_results.html', result=result)

if __name__ == '__main__':
    print("🐄 INDIAN CATTLE BREED RECOGNITION WEB APP")
    print("=" * 55)
    print(f"🧠 Model Status: {'✅ Loaded' if classifier.is_loaded else '❌ Not Available'}")
    if classifier.is_loaded:
        print(f"🎯 Model Accuracy: {classifier.model_accuracy:.1f}%")
        print(f"🏷️  Classes: {len(classifier.class_names)} Indian breeds")
    else:
        print("💡 Run 'python indian_cattle_finetuning.py' first to train the model")
    print(f"📱 Starting server...")
    print(f"🌐 Visit: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)