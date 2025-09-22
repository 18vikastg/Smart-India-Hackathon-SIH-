#!/usr/bin/env python3
"""
Test the trained Indian Cattle Breed Recognition model
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import os
import json

def test_indian_model():
    """Test the trained Indian cattle model"""
    
    print("🧪 TESTING INDIAN CATTLE BREED MODEL")
    print("=" * 50)
    
    model_path = 'best_indian_cattle_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please run 'python indian_cattle_finetuning.py' first to train the model")
        return
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        checkpoint = torch.load(model_path, map_location=device)
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        accuracy = checkpoint.get('accuracy', 0.0)
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Training Accuracy: {accuracy:.2f}%")
        print(f"🏷️  Number of breeds: {num_classes}")
        
        # Build model architecture
        model = resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"\n🐄 Indian Cattle Breeds in the model:")
        indian_breeds = []
        for i, breed in enumerate(class_names):
            print(f"  {i+1:2d}. {breed}")
            if not any(x in breed for x in ['Holstein', 'Jersey', 'Ayrshire', 'Brown_Swiss', 'Guernsey', 'Red_Dane']):
                indian_breeds.append(breed)
        
        print(f"\n🇮🇳 Authentic Indian breeds: {len(indian_breeds)} out of {num_classes}")
        print("Top Indian breeds:", indian_breeds[:10])
        
        # Test with sample images if available
        sample_images = []
        for filename in os.listdir('uploads'):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join('uploads', filename))
        
        if sample_images:
            print(f"\n📸 Testing with {len(sample_images)} sample images:")
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            for i, img_path in enumerate(sample_images[:3]):  # Test first 3 images
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        top3_prob, top3_indices = torch.topk(probabilities, 3)
                    
                    print(f"\n  📷 {os.path.basename(img_path)}:")
                    for j, (prob, idx) in enumerate(zip(top3_prob, top3_indices)):
                        breed = class_names[idx.item()]
                        confidence = prob.item() * 100
                        print(f"    {j+1}. {breed}: {confidence:.1f}%")
                
                except Exception as e:
                    print(f"  ❌ Error testing {img_path}: {e}")
        
        print(f"\n✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    test_indian_model()