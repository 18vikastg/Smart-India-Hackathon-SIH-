# Enhanced Cattle Classifier with Ensemble Learning and Multi-Task Classification
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm
import numpy as np
from datetime import datetime

class AdvancedCattleEnsemble(nn.Module):
    """
    Advanced ensemble model combining ResNet50, EfficientNet, and Vision Transformer
    for improved cattle breed recognition with multi-task learning capabilities.
    """
    
    def __init__(self, num_breeds=41, num_gender=2, num_age=3):
        super(AdvancedCattleEnsemble, self).__init__()
        
        # Model 1: ResNet50 (proven performer from our base system)
        self.resnet50 = models.resnet50(pretrained=True)
        resnet_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()  # Remove final layer
        
        # Model 2: EfficientNet-B4 (efficiency + accuracy balance)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        efficientnet_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove final layer
        
        # Model 3: Vision Transformer (attention mechanism for fine details)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_features = self.vit.num_features
        
        # Feature dimensions
        self.resnet_features = resnet_features
        self.efficientnet_features = efficientnet_features  
        self.vit_features = vit_features
        self.combined_features = resnet_features + efficientnet_features + vit_features
        
        # Individual breed classifiers for each model
        self.resnet_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_breeds)
        )
        
        self.efficientnet_classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(efficientnet_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_breeds)
        )
        
        self.vit_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(vit_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_breeds)
        )
        
        # Ensemble fusion network
        self.breed_fusion = nn.Sequential(
            nn.Linear(num_breeds * 3, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_breeds)
        )
        
        # Multi-task learning heads using combined features
        self.gender_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_gender)
        )
        
        self.age_head = nn.Sequential(
            nn.Linear(self.combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_age)
        )
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(self.combined_features, self.combined_features // 4),
            nn.ReLU(),
            nn.Linear(self.combined_features // 4, self.combined_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features from all models
        resnet_features = self.extract_resnet_features(x)
        efficientnet_features = self.extract_efficientnet_features(x)
        vit_features = self.extract_vit_features(x)
        
        # Individual predictions for ensemble consistency
        resnet_pred = self.resnet_classifier(resnet_features)
        efficientnet_pred = self.efficientnet_classifier(efficientnet_features)
        vit_pred = self.vit_classifier(vit_features)
        
        # Combine predictions for ensemble
        combined_preds = torch.cat([resnet_pred, efficientnet_pred, vit_pred], dim=1)
        ensemble_breed_output = self.breed_fusion(combined_preds)
        
        # Combine features for multi-task learning
        combined_features = torch.cat([resnet_features, efficientnet_features, vit_features], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.feature_attention(combined_features)
        attended_features = combined_features * attention_weights
        
        # Multi-task outputs
        gender_output = self.gender_head(attended_features)
        age_output = self.age_head(attended_features)
        
        # Temperature scaling for calibrated confidence
        calibrated_breed = ensemble_breed_output / self.temperature
        
        return {
            'breed': calibrated_breed,
            'gender': gender_output,
            'age': age_output,
            'individual_preds': [resnet_pred, efficientnet_pred, vit_pred],
            'features': {
                'resnet': resnet_features,
                'efficientnet': efficientnet_features,
                'vit': vit_features,
                'combined': attended_features
            }
        }
    
    def extract_resnet_features(self, x):
        """Extract features from ResNet50 backbone"""
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def extract_efficientnet_features(self, x):
        """Extract features from EfficientNet backbone"""
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = torch.flatten(x, 1)
        return x
    
    def extract_vit_features(self, x):
        """Extract features from Vision Transformer"""
        return self.vit(x)
    
    def get_prediction_confidence(self, outputs):
        """Calculate calibrated confidence scores"""
        breed_probs = torch.softmax(outputs['breed'], dim=1)
        max_prob = torch.max(breed_probs, dim=1)[0]
        
        # Entropy-based uncertainty
        entropy = -torch.sum(breed_probs * torch.log(breed_probs + 1e-8), dim=1)
        normalized_entropy = entropy / np.log(breed_probs.size(1))
        uncertainty = normalized_entropy
        
        # Calibrated confidence (higher is more confident)
        calibrated_confidence = max_prob * (1 - uncertainty)
        
        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'uncertainty': uncertainty,
            'calibrated_confidence': calibrated_confidence
        }


class EnsembleTrainer:
    """
    Training class for the advanced ensemble model with multi-task learning
    """
    
    def __init__(self, model, device, breed_weight=0.6, gender_weight=0.2, age_weight=0.1, consistency_weight=0.1):
        self.model = model
        self.device = device
        
        # Loss function weights
        self.breed_weight = breed_weight
        self.gender_weight = gender_weight
        self.age_weight = age_weight
        self.consistency_weight = consistency_weight
        
        # Loss functions
        self.breed_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def compute_multi_task_loss(self, outputs, targets):
        """
        Compute combined loss for multi-task learning with ensemble consistency
        """
        # Primary task losses
        breed_loss = self.breed_criterion(outputs['breed'], targets['breed'])
        
        # Multi-task losses (if targets available)
        gender_loss = torch.tensor(0.0, device=self.device)
        age_loss = torch.tensor(0.0, device=self.device)
        
        if 'gender' in targets and targets['gender'] is not None:
            gender_loss = self.gender_criterion(outputs['gender'], targets['gender'])
            
        if 'age' in targets and targets['age'] is not None:
            age_loss = self.age_criterion(outputs['age'], targets['age'])
        
        # Ensemble consistency loss
        consistency_loss = self.compute_consistency_loss(outputs['individual_preds'])
        
        # Combined weighted loss
        total_loss = (self.breed_weight * breed_loss + 
                     self.gender_weight * gender_loss + 
                     self.age_weight * age_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'breed_loss': breed_loss.item(),
            'gender_loss': gender_loss.item(),
            'age_loss': age_loss.item(),
            'consistency_loss': consistency_loss.item()
        }
    
    def compute_consistency_loss(self, individual_preds):
        """
        Compute consistency loss between individual model predictions
        """
        consistency_loss = 0.0
        num_models = len(individual_preds)
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # KL divergence between softmax distributions
                pred_i = torch.log_softmax(individual_preds[i], dim=1)
                pred_j = torch.softmax(individual_preds[j], dim=1)
                consistency_loss += self.kl_div(pred_i, pred_j)
        
        # Average over all pairs
        num_pairs = (num_models * (num_models - 1)) / 2
        return consistency_loss / num_pairs if num_pairs > 0 else consistency_loss
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """
        Train the model for one epoch
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {'breed': 0.0, 'gender': 0.0, 'age': 0.0, 'consistency': 0.0}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'breed': batch['breed'].to(self.device),
                'gender': batch.get('gender', torch.tensor([]).to(self.device)),
                'age': batch.get('age', torch.tensor([]).to(self.device))
            }
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.compute_multi_task_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for key in loss_components:
                if key + '_loss' in loss_dict:
                    loss_components[key] += loss_dict[key + '_loss']
        
        # Step scheduler if provided
        if scheduler:
            scheduler.step()
        
        # Average losses
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self, val_loader):
        """
        Validate the model for one epoch
        """
        self.model.eval()
        total_loss = 0.0
        correct_breed = 0
        correct_gender = 0
        correct_age = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                targets = {
                    'breed': batch['breed'].to(self.device),
                    'gender': batch.get('gender', torch.tensor([]).to(self.device)),
                    'age': batch.get('age', torch.tensor([]).to(self.device))
                }
                
                outputs = self.model(images)
                loss, _ = self.compute_multi_task_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracies
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Breed accuracy
                _, breed_pred = torch.max(outputs['breed'], 1)
                correct_breed += (breed_pred == targets['breed']).sum().item()
                
                # Gender accuracy (if available)
                if len(targets['gender']) > 0:
                    _, gender_pred = torch.max(outputs['gender'], 1)
                    correct_gender += (gender_pred == targets['gender']).sum().item()
                
                # Age accuracy (if available)
                if len(targets['age']) > 0:
                    _, age_pred = torch.max(outputs['age'], 1)
                    correct_age += (age_pred == targets['age']).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        breed_accuracy = correct_breed / total_samples
        gender_accuracy = correct_gender / total_samples if correct_gender > 0 else 0.0
        age_accuracy = correct_age / total_samples if correct_age > 0 else 0.0
        
        return avg_loss, {
            'breed_accuracy': breed_accuracy,
            'gender_accuracy': gender_accuracy,
            'age_accuracy': age_accuracy
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCattleEnsemble(num_breeds=41, num_gender=2, num_age=3).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    outputs = model(dummy_input)
    
    print(f"Model initialized successfully!")
    print(f"Breed output shape: {outputs['breed'].shape}")
    print(f"Gender output shape: {outputs['gender'].shape}")
    print(f"Age output shape: {outputs['age'].shape}")
    print(f"Individual predictions: {len(outputs['individual_preds'])}")
    
    # Test confidence calculation
    confidence_metrics = model.get_prediction_confidence(outputs)
    print(f"Confidence metrics calculated: {list(confidence_metrics.keys())}")
    
    print("\n✅ Advanced Ensemble Model ready for training!")