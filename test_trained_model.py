#!/usr/bin/env python3
"""
Demonstrate the fine-tuned cattle breed classifier
Load and test the trained model on new images
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import os

class FineTunedCattleClassifier:
    """Load and use the fine-tuned cattle breed model"""
    
    def __init__(self, model_path='best_cattle_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"🐄 Fine-tuned Cattle Classifier Loaded!")
        print(f"📊 Model Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"🏷️  Classes: {self.class_names}")
    
    def predict(self, image_path):
        """Predict cattle breed from image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # Get top predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            results = []
            for i in range(3):
                breed = self.class_names[top3_idx[i]]
                confidence = top3_prob[i].item() * 100
                results.append((breed, confidence))
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []

def test_trained_model():
    """Test the fine-tuned model on sample images"""
    
    print("🧪 TESTING FINE-TUNED CATTLE CLASSIFIER")
    print("=" * 50)
    
    # Load classifier
    try:
        classifier = FineTunedCattleClassifier()
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("💡 Make sure 'best_cattle_model.pth' exists (run fine-tuning first)")
        return
    
    # Test on some sample images from each breed
    test_images = []
    
    # Find sample images from each breed
    breeds_dir = "Cattle Breeds"
    for breed in classifier.class_names:
        breed_dir = os.path.join(breeds_dir, breed)
        if os.path.exists(breed_dir):
            images = [f for f in os.listdir(breed_dir) if f.endswith('.jpg')][:2]
            for img in images:
                test_images.append(os.path.join(breed_dir, img))
    
    print(f"\n🔍 Testing on {len(test_images)} sample images:")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = 0
    
    for image_path in test_images[:10]:  # Test first 10 images
        print(f"\n📸 Testing: {os.path.basename(image_path)}")
        
        # Get actual breed from path
        actual_breed = os.path.basename(os.path.dirname(image_path))
        
        # Make prediction
        predictions = classifier.predict(image_path)
        
        if predictions:
            predicted_breed, confidence = predictions[0]
            
            print(f"   🏷️  Actual: {actual_breed}")
            print(f"   🎯 Predicted: {predicted_breed} ({confidence:.1f}%)")
            
            # Check if prediction is correct
            if predicted_breed == actual_breed:
                print(f"   ✅ CORRECT!")
                correct_predictions += 1
            else:
                print(f"   ❌ Incorrect")
                
            # Show top 3 predictions
            print(f"   📊 Top 3:")
            for i, (breed, conf) in enumerate(predictions):
                print(f"      {i+1}. {breed}: {conf:.1f}%")
            
            total_predictions += 1
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n🎯 Test Results:")
        print(f"   Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"   Model Performance: {'Excellent' if accuracy > 80 else 'Good' if accuracy > 60 else 'Needs Improvement'}")
    
    print(f"\n✨ Model is ready for deployment!")
    print(f"   You can now use this classifier in your web app")
    print(f"   or integrate it into other applications!")

if __name__ == "__main__":
    test_trained_model()