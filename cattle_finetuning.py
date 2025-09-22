#!/usr/bin/env python3
"""
Complete Cattle Breed Fine-tuning System
Fine-tune ResNet50 on your European cattle dataset: 1,208 images across 5 breeds
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
from datetime import datetime

class CattleDataset(Dataset):
    """Custom dataset for cattle breed images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CattleBreedFineTuner:
    """Complete fine-tuning pipeline for cattle breed classification"""
    
    def __init__(self, dataset_path, batch_size=16, learning_rate=0.001):
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing Fine-tuning System")
        print(f"📱 Device: {self.device}")
        print(f"📊 Batch Size: {batch_size}")
        print(f"📈 Learning Rate: {learning_rate}")
        
        # Define class names (your 5 European breeds)
        self.class_names = [
            'Ayrshire cattle',
            'Brown Swiss cattle', 
            'Holstein Friesian cattle',
            'Jersey cattle',
            'Red Dane cattle'
        ]
        self.num_classes = len(self.class_names)
        
        # Define data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_dataset(self):
        """Load and organize the cattle dataset"""
        print("\n📂 Loading Dataset...")
        
        all_image_paths = []
        all_labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.dataset_path / class_name
            
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_name} directory not found!")
                continue
            
            # Get all jpg images in the class directory
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            
            print(f"📸 {class_name}: {len(image_files)} images")
            
            for image_path in image_files:
                all_image_paths.append(str(image_path))
                all_labels.append(class_idx)
        
        print(f"\n📊 Total Dataset: {len(all_image_paths)} images across {self.num_classes} breeds")
        
        # Split into train/validation sets (80/20 split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        print(f"🎯 Training Set: {len(train_paths)} images")
        print(f"🎯 Validation Set: {len(val_paths)} images")
        
        return train_paths, val_paths, train_labels, val_labels
    
    def create_data_loaders(self, train_paths, val_paths, train_labels, val_labels):
        """Create PyTorch data loaders"""
        print("\n🔄 Creating Data Loaders...")
        
        train_dataset = CattleDataset(train_paths, train_labels, self.train_transform)
        val_dataset = CattleDataset(val_paths, val_labels, self.val_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_model(self):
        """Create and configure the ResNet50 model for fine-tuning"""
        print("\n🧠 Creating Model...")
        
        # Load pretrained ResNet50
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers (feature extractor)
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final layer for our classes
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        # Unfreeze final layers for fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        
        model = model.to(self.device)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total Parameters: {total_params:,}")
        print(f"🎯 Trainable Parameters: {trainable_params:,}")
        print(f"🔒 Frozen Parameters: {total_params - trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={100*correct/total:.1f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def fine_tune_model(self, epochs=10):
        """Complete fine-tuning pipeline"""
        print(f"\n🚀 Starting Fine-tuning Process ({epochs} epochs)")
        print("=" * 60)
        
        # Load dataset
        train_paths, val_paths, train_labels, val_labels = self.load_dataset()
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_paths, val_paths, train_labels, val_labels
        )
        
        # Create model
        model = self.create_model()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        print(f"\n🎯 Training Configuration:")
        print(f"   Loss Function: CrossEntropyLoss")
        print(f"   Optimizer: Adam (lr={self.learning_rate})")
        print(f"   Scheduler: StepLR (step=5, gamma=0.5)")
        print(f"   Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # Training phase
            print("🏋️  Training...")
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            print("🔍 Validating...")
            val_loss, val_acc, val_preds, val_true = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Save metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"🎉 New best validation accuracy: {val_acc:.2f}%")
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }, 'best_cattle_model.pth')
                print("💾 Model saved as 'best_cattle_model.pth'")
        
        total_time = time.time() - start_time
        
        print(f"\n🏁 Training Complete!")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
        
        # Final classification report
        print(f"\n📋 Final Classification Report:")
        report = classification_report(val_true, val_preds, 
                                     target_names=self.class_names, 
                                     digits=3)
        print(report)
        
        return model, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }

def demonstrate_finetuning():
    """Demonstrate the complete fine-tuning process"""
    
    print("🐄 CATTLE BREED FINE-TUNING DEMONSTRATION")
    print("=" * 60)
    print("Fine-tuning ResNet50 on European Cattle Dataset")
    print("Dataset: 1,208 images across 5 breeds")
    print("Method: Transfer Learning with Layer Freezing")
    
    # Dataset path
    dataset_path = "Cattle Breeds"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Make sure you have the 'Cattle Breeds' directory with:")
        print("   - Ayrshire cattle/")
        print("   - Brown Swiss cattle/")
        print("   - Holstein Friesian cattle/")
        print("   - Jersey cattle/")
        print("   - Red Dane cattle/")
        return
    
    # Initialize fine-tuner
    finetuner = CattleBreedFineTuner(
        dataset_path=dataset_path,
        batch_size=8,  # Smaller batch size for CPU
        learning_rate=0.001
    )
    
    try:
        # Run fine-tuning (reduced epochs for demonstration)
        model, history = finetuner.fine_tune_model(epochs=3)
        
        print(f"\n🎯 Fine-tuning Summary:")
        print(f"✅ Model successfully fine-tuned")
        print(f"✅ Best accuracy: {history['best_val_acc']:.2f}%")
        print(f"✅ Model saved for deployment")
        
        print(f"\n📈 Training Progress:")
        for i, (train_acc, val_acc) in enumerate(zip(history['train_accuracies'], history['val_accuracies'])):
            print(f"   Epoch {i+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%")
        
        print(f"\n🚀 Model Ready for Deployment!")
        print(f"   Load with: torch.load('best_cattle_model.pth')")
        print(f"   Classes: {finetuner.class_names}")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        print("💡 This might be due to insufficient memory or missing dependencies")

if __name__ == "__main__":
    demonstrate_finetuning()