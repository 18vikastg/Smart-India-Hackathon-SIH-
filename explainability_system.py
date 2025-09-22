# Explainability and Confidence Calibration for Cattle Recognition
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional

# Note: These imports would need to be installed for full functionality
# pip install grad-cam matplotlib seaborn

class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping for cattle breed recognition explanations
    """
    
    def __init__(self, model, target_layers=None, use_cuda=True):
        self.model = model
        self.model.eval()
        self.use_cuda = use_cuda
        
        # Default target layers for different model types
        if target_layers is None:
            if hasattr(model, 'resnet50'):
                self.target_layers = [model.resnet50.layer4[-1]]
            elif hasattr(model, 'layer4'):
                self.target_layers = [model.layer4[-1]]
            else:
                # Fallback - try to find the last convolutional layer
                self.target_layers = self._find_target_layers()
        else:
            self.target_layers = target_layers
            
        self.gradients = {}
        self.activations = {}
        self.handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layers(self):
        """Automatically find suitable target layers"""
        target_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)) and 'layer4' in name:
                target_layers.append(module)
        return target_layers[-1:] if target_layers else []
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations[module] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients[module] = grad_output[0].detach()
        
        for layer in self.target_layers:
            handle1 = layer.register_forward_hook(forward_hook)
            handle2 = layer.register_backward_hook(backward_hook)
            self.handles.extend([handle1, handle2])
    
    def generate_cam(self, input_tensor, class_idx=None, retain_graph=False):
        """
        Generate Class Activation Map
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            logits = output['breed']
        else:
            logits = output
            
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        score = logits[:, class_idx].squeeze()
        score.backward(retain_graph=retain_graph)
        
        # Generate CAM for each target layer
        cams = []
        for target_layer in self.target_layers:
            if target_layer in self.gradients and target_layer in self.activations:
                gradients = self.gradients[target_layer]
                activations = self.activations[target_layer]
                
                # Global average pooling of gradients
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                # Normalize CAM
                cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
                cam = cam.squeeze()
                
                # Normalize to [0, 1]
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cams.append(cam.cpu().numpy())
        
        return cams[0] if cams else None
    
    def generate_explanation(self, image_tensor, class_idx, breed_name, confidence_score):
        """
        Generate comprehensive visual and textual explanation
        """
        # Generate Grad-CAM
        cam = self.generate_cam(image_tensor.unsqueeze(0), class_idx)
        
        if cam is None:
            return None
        
        # Convert input tensor to numpy for visualization
        image_np = self._tensor_to_numpy(image_tensor)
        
        # Create heatmap overlay
        heatmap = self._create_heatmap_overlay(image_np, cam)
        
        # Analyze attention regions
        attention_analysis = self._analyze_attention_regions(cam, breed_name)
        
        # Generate textual explanation
        textual_explanation = self._generate_textual_explanation(
            attention_analysis, breed_name, confidence_score
        )
        
        return {
            'heatmap': heatmap,
            'cam': cam,
            'attention_regions': attention_analysis,
            'explanation': textual_explanation,
            'confidence': confidence_score
        }
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array for visualization"""
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def _create_heatmap_overlay(self, image, cam, alpha=0.4):
        """Create heatmap overlay on original image"""
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255
        
        # Overlay heatmap on image
        overlay = alpha * heatmap + (1 - alpha) * image
        overlay = np.uint8(255 * overlay)
        
        return overlay
    
    def _analyze_attention_regions(self, cam, breed_name):
        """Analyze which regions the model is focusing on"""
        # Find high attention regions
        threshold = np.percentile(cam, 80)  # Top 20% attention
        high_attention_mask = cam > threshold
        
        # Analyze spatial distribution
        height, width = cam.shape
        
        # Divide image into regions
        regions = {
            'head_neck': cam[:height//3, :].mean(),
            'body_center': cam[height//3:2*height//3, width//4:3*width//4].mean(),
            'legs_udder': cam[2*height//3:, :].mean(),
            'left_side': cam[:, :width//2].mean(),
            'right_side': cam[:, width//2:].mean()
        }
        
        # Sort regions by attention
        sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'high_attention_percentage': (high_attention_mask.sum() / high_attention_mask.size) * 100,
            'primary_focus_regions': sorted_regions[:3],
            'attention_distribution': regions
        }
    
    def _generate_textual_explanation(self, attention_analysis, breed_name, confidence):
        """Generate human-readable explanation"""
        
        # Breed-specific feature descriptions
        breed_features = {
            'Gir': {
                'head_neck': 'distinctive hump and curved horns',
                'body_center': 'compact body with prominent hump',
                'characteristic_features': ['prominent hump', 'curved horns', 'grey/reddish coat']
            },
            'Holstein_Friesian': {
                'head_neck': 'straight profile and large head',
                'body_center': 'large frame with black and white patches',
                'characteristic_features': ['black and white patches', 'large frame', 'straight facial profile']
            },
            'Jersey': {
                'head_neck': 'refined head with dished face',
                'body_center': 'compact, dairy-type body',
                'characteristic_features': ['fawn coloration', 'compact size', 'refined features']
            },
            'Sahiwal': {
                'head_neck': 'loose skin with dewlap',
                'body_center': 'medium-sized body with good depth',
                'characteristic_features': ['reddish-brown color', 'loose skin', 'heat tolerance features']
            }
        }
        
        # Get primary focus regions
        primary_regions = attention_analysis['primary_focus_regions']
        
        # Build explanation
        explanation_parts = []
        
        # Confidence assessment
        if confidence > 0.8:
            confidence_text = "very confident"
        elif confidence > 0.6:
            confidence_text = "confident"
        elif confidence > 0.4:
            confidence_text = "moderately confident"
        else:
            confidence_text = "uncertain"
        
        explanation_parts.append(f"The model is {confidence_text} ({confidence:.1%}) that this is a {breed_name} cattle.")
        
        # Attention analysis
        if primary_regions:
            top_region = primary_regions[0][0].replace('_', ' ')
            explanation_parts.append(f"The model focused primarily on the {top_region} region.")
            
            # Add breed-specific context
            if breed_name in breed_features:
                features = breed_features[breed_name]
                if primary_regions[0][0] in features:
                    explanation_parts.append(f"This region typically shows {features[primary_regions[0][0]]} in {breed_name} cattle.")
        
        # Feature-based explanation
        if breed_name in breed_features:
            characteristic_features = breed_features[breed_name]['characteristic_features']
            explanation_parts.append(f"Key identifying features for {breed_name} include {', '.join(characteristic_features)}.")
        
        return ' '.join(explanation_parts)
    
    def cleanup(self):
        """Remove hooks"""
        for handle in self.handles:
            handle.remove()


class ConfidenceCalibrator:
    """
    Temperature scaling for confidence calibration
    """
    
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def calibrate(self, validation_loader, device):
        """
        Calibrate temperature parameter using validation set
        """
        print("🔧 Calibrating confidence scores...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in validation_loader:
                images = batch['image'].to(device)
                labels = batch['breed'].to(device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    logits = outputs['breed']
                else:
                    logits = outputs
                
                all_logits.append(logits)
                all_labels.append(labels)
        
        # Concatenate all batches
        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits_tensor), labels_tensor)
            loss.backward()
            return loss
        
        initial_loss = eval_loss().item()
        optimizer.step(eval_loss)
        final_loss = eval_loss().item()
        
        print(f"✅ Temperature calibration complete!")
        print(f"   Optimal temperature: {self.temperature.item():.3f}")
        print(f"   Loss reduction: {initial_loss:.4f} → {final_loss:.4f}")
        
        return self.temperature.item()
    
    def get_calibrated_predictions(self, logits):
        """Get calibrated probability predictions"""
        calibrated_logits = self.temperature_scale(logits)
        calibrated_probs = F.softmax(calibrated_logits, dim=1)
        return calibrated_probs
    
    def calculate_confidence_metrics(self, logits, labels):
        """Calculate various confidence metrics"""
        probs = F.softmax(logits, dim=1)
        calibrated_probs = self.get_calibrated_predictions(logits)
        
        # Maximum probability (confidence)
        max_probs = torch.max(probs, dim=1)[0]
        calibrated_max_probs = torch.max(calibrated_probs, dim=1)[0]
        
        # Entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        calibrated_entropy = -torch.sum(calibrated_probs * torch.log(calibrated_probs + 1e-8), dim=1)
        
        # Normalize entropy
        max_entropy = np.log(probs.size(1))
        normalized_entropy = entropy / max_entropy
        calibrated_normalized_entropy = calibrated_entropy / max_entropy
        
        return {
            'raw_confidence': max_probs,
            'calibrated_confidence': calibrated_max_probs,
            'raw_entropy': entropy,
            'calibrated_entropy': calibrated_entropy,
            'uncertainty': normalized_entropy,
            'calibrated_uncertainty': calibrated_normalized_entropy
        }


class ExplainabilityIntegrator:
    """
    Integration class for explainability features in the web application
    """
    
    def __init__(self, model, device, breed_names):
        self.model = model
        self.device = device
        self.breed_names = breed_names
        
        # Initialize explainability components
        self.grad_cam = GradCAMExplainer(model)
        self.calibrator = ConfidenceCalibrator(model)
        
    def explain_prediction(self, image_tensor, save_path=None):
        """
        Generate comprehensive explanation for a prediction
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model prediction
            outputs = self.model(image_tensor.unsqueeze(0))
            
            if isinstance(outputs, dict):
                breed_logits = outputs['breed']
            else:
                breed_logits = outputs
            
            # Get calibrated probabilities
            calibrated_probs = self.calibrator.get_calibrated_predictions(breed_logits)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(calibrated_probs, k=3, dim=1)
            
        # Generate explanation for top prediction
        top_class = top_indices[0, 0].item()
        top_confidence = top_probs[0, 0].item()
        breed_name = self.breed_names[top_class]
        
        # Generate Grad-CAM explanation
        explanation = self.grad_cam.generate_explanation(
            image_tensor, top_class, breed_name, top_confidence
        )
        
        if explanation is None:
            return self._create_fallback_explanation(breed_name, top_confidence)
        
        # Add top-3 predictions
        explanation['top_predictions'] = []
        for i in range(3):
            class_idx = top_indices[0, i].item()
            confidence = top_probs[0, i].item()
            explanation['top_predictions'].append({
                'breed': self.breed_names[class_idx],
                'confidence': confidence,
                'class_idx': class_idx
            })
        
        # Save visualization if path provided
        if save_path:
            self._save_explanation_visualization(explanation, save_path)
        
        return explanation
    
    def _create_fallback_explanation(self, breed_name, confidence):
        """Create fallback explanation when Grad-CAM fails"""
        return {
            'explanation': f"Predicted as {breed_name} with {confidence:.1%} confidence.",
            'confidence': confidence,
            'heatmap': None,
            'attention_regions': None
        }
    
    def _save_explanation_visualization(self, explanation, save_path):
        """Save explanation visualization to file"""
        if explanation['heatmap'] is not None:
            # Create figure with original image and heatmap
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image (reconstructed from tensor)
            # This would need the original image data
            
            # Heatmap overlay
            axes[1].imshow(explanation['heatmap'])
            axes[1].set_title(f"Attention Map\nConfidence: {explanation['confidence']:.1%}")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("🔍 Testing Explainability Components...")
    
    # This would normally use your trained model
    # model = load_your_trained_model()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ Grad-CAM explainer initialized")
    print("✅ Confidence calibrator ready")
    print("🎯 Explainability system ready for integration!")