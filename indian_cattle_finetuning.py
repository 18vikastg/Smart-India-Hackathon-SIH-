#!/usr/bin/env python3
"""
Fine-tuning ResNet50 for Indian Bovine Breeds Classification
Using authentic Indian cattle breeds dataset from Kaggle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class IndianCattleDataset(Dataset):
    """Dataset class for Indian Bovine Breeds"""
    
    def __init__(self, dataframe, dataset_path, transform=None):
        self.df = dataframe
        self.dataset_path = dataset_path
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Fix path separators (dataset uses Windows paths)
        image_path = os.path.join(
            self.dataset_path, 
            "Indian_bovine_breeds", 
            "Indian_bovine_breeds",
            row['path'].replace('\\', os.sep)
        )
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get breed label
            breed = row['breed']
            
            return image, breed
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, row['breed']

class IndianCattleClassifier:
    """Fine-tuned ResNet50 for Indian Cattle Breeds"""
    
    def __init__(self, num_classes=41):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = []
        self.model = None
        self.model_accuracy = 0.0
        
        print(f"🔧 Using device: {self.device}")
        
    def build_model(self):
        """Build ResNet50 model with custom classifier"""
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Freeze early layers (fine-tuning approach)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        # Unfreeze the last few layers for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        self.model = self.model.to(self.device)
        print(f"✅ Model built with {self.num_classes} classes")
        
    def prepare_data(self, dataset_path, metadata_path, test_size=0.2, val_size=0.1):
        """Prepare training, validation, and test datasets"""
        
        print("📋 Loading metadata...")
        df = pd.read_csv(metadata_path)
        
        # Get unique breeds and create label mapping
        unique_breeds = sorted(df['breed'].unique())
        self.class_names = unique_breeds
        breed_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
        
        # Add numeric labels
        df['label'] = df['breed'].map(breed_to_idx)
        
        print(f"🏷️  Found {len(unique_breeds)} breeds:")
        for i, breed in enumerate(unique_breeds):
            count = df[df['breed'] == breed].shape[0]
            print(f"  {i+1:2d}. {breed}: {count} images")
        
        # Split the data
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['breed'], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, stratify=train_df['breed'], random_state=42
        )
        
        print(f"\n📊 Data splits:")
        print(f"  🔵 Training: {len(train_df)} images")
        print(f"  🟡 Validation: {len(val_df)} images")
        print(f"  🟢 Test: {len(test_df)} images")
        
        # Data transformations
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = IndianCattleDataset(train_df, dataset_path, train_transform)
        val_dataset = IndianCattleDataset(val_df, dataset_path, val_transform)
        test_dataset = IndianCattleDataset(test_df, dataset_path, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=25, learning_rate=0.001):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_path = 'best_indian_cattle_model.pth'
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (images, breeds) in enumerate(train_loader):
                # Convert breed names to indices
                breed_indices = torch.tensor([self.class_names.index(breed) for breed in breeds])
                
                images = images.to(self.device)
                labels = breed_indices.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx:3d}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            train_acc = 100 * correct_train / total_train
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for images, breeds in val_loader:
                    breed_indices = torch.tensor([self.class_names.index(breed) for breed in breeds])
                    
                    images = images.to(self.device)
                    labels = breed_indices.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_acc = 100 * correct_val / total_val
            val_loss = val_loss / len(val_loader)
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'class_names': self.class_names,
                    'num_classes': self.num_classes,
                    'accuracy': val_acc
                }, best_model_path)
                print(f"  💾 New best model saved! Validation accuracy: {val_acc:.2f}%")
            
            # Print epoch results
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        self.model_accuracy = best_val_acc
        print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        print("\n🧪 Evaluating on test set...")
        
        with torch.no_grad():
            for images, breeds in test_loader:
                breed_indices = torch.tensor([self.class_names.index(breed) for breed in breeds])
                
                images = images.to(self.device)
                labels = breed_indices.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100 * correct / total
        print(f"✅ Test Accuracy: {test_accuracy:.2f}%")
        
        # Generate classification report
        breed_names = [self.class_names[i] for i in range(len(self.class_names))]
        report = classification_report(all_labels, all_predictions, 
                                     target_names=breed_names, 
                                     zero_division=0)
        print(f"\n📊 Classification Report:")
        print(report)
        
        return test_accuracy, all_predictions, all_labels

def main():
    """Main training function"""
    
    print("🐄 INDIAN BOVINE BREEDS CLASSIFICATION")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset paths
    dataset_path = "/home/vikas/.cache/kagglehub/datasets/lukex9442/indian-bovine-breeds/versions/5"
    metadata_path = os.path.join(dataset_path, "bovine_breeds_metadata.csv")
    
    # Initialize classifier
    classifier = IndianCattleClassifier(num_classes=41)
    classifier.build_model()
    
    # Prepare data
    train_loader, val_loader, test_loader = classifier.prepare_data(
        dataset_path, metadata_path, test_size=0.2, val_size=0.1
    )
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = classifier.train(
        train_loader, val_loader, epochs=25, learning_rate=0.001
    )
    
    # Evaluate model
    test_accuracy, predictions, labels = classifier.evaluate(test_loader)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'test_accuracy': test_accuracy,
        'class_names': classifier.class_names,
        'num_breeds': len(classifier.class_names)
    }
    
    with open('indian_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"   📊 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"   🏷️  Total Breeds: {len(classifier.class_names)}")
    print(f"   💾 Model saved as: best_indian_cattle_model.pth")
    print(f"   📈 History saved as: indian_training_history.json")

if __name__ == "__main__":
    main()