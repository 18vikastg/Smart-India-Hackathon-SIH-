# Advanced Data Augmentation for Cattle Breed Recognition
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image, ImageEnhance, ImageFilter
import os
from collections import defaultdict

class CattleSpecificAugmentation:
    """
    Breed-specific augmentation techniques for Indian cattle breeds
    """
    
    def __init__(self):
        self.breed_augmentation_map = {
            'Gir': self.gir_augmentations(),
            'Sahiwal': self.sahiwal_augmentations(),
            'Red_Sindhi': self.red_sindhi_augmentations(),
            'Tharparkar': self.tharparkar_augmentations(),
            'Holstein_Friesian': self.holstein_augmentations(),
            'Jersey': self.jersey_augmentations(),
            'Brown_Swiss': self.brown_swiss_augmentations(),
            'Ayrshire': self.ayrshire_augmentations(),
            # Add more breeds as needed
        }
        
    def gir_augmentations(self):
        """Augmentations specific to Gir cattle - emphasize hump, horns, and coat variations"""
        return A.Compose([
            # Color variations for Gir's diverse coat colors (white, grey, reddish)
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            
            # Shadow and lighting variations (tropical climate simulation)
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, p=0.3),
            
            # Perspective changes to emphasize hump structure
            A.Perspective(scale=(0.05, 0.15), p=0.6),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.8),
            
            # Environmental variations
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.Blur(blur_limit=3, p=0.3),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        ])
    
    def sahiwal_augmentations(self):
        """Augmentations for Sahiwal cattle - focus on reddish-brown coloration"""
        return A.Compose([
            # Enhance reddish-brown color characteristics
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=10, p=0.6),
            
            # Heat stress simulation (high temperature environments)
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # Body structure emphasis
            A.Perspective(scale=(0.02, 0.1), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=15, p=0.7),
            
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        ])
    
    def holstein_augmentations(self):
        """Augmentations for Holstein - emphasize black and white pattern variations"""
        return A.Compose([
            # High contrast for black/white patterns
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.5, p=0.8),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
            
            # Pattern preservation with minor color variations
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
            
            # Size variations (Holstein are large cattle)
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.6),
            
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        ])
    
    def jersey_augmentations(self):
        """Augmentations for Jersey cattle - focus on fawn/light brown coloration"""
        return A.Compose([
            # Fawn color variations
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.8),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.3, hue=0.08, p=0.7),
            
            # Compact size emphasis
            A.RandomScale(scale_limit=(-0.1, 0.1), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=12, p=0.6),
            
            # Environmental variations
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.4),
            
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ])
    
    def get_breed_augmentation(self, breed_name):
        """Get breed-specific augmentation pipeline"""
        return self.breed_augmentation_map.get(breed_name, self.default_augmentation())
    
    def default_augmentation(self):
        """Default augmentation for breeds without specific pipelines"""
        return A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ])


class SyntheticDataGenerator:
    """
    Generate synthetic cattle images using advanced augmentation techniques
    """
    
    def __init__(self, augmentation_pipeline):
        self.augmentation = augmentation_pipeline
        
    def generate_synthetic_images(self, base_images, breed_name, target_count, output_dir):
        """
        Generate synthetic images for a specific breed to balance dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        breed_augmentation = self.augmentation.get_breed_augmentation(breed_name)
        
        synthetic_count = 0
        generated_images = []
        
        while synthetic_count < target_count:
            # Select random base image
            base_image_path = random.choice(base_images)
            
            try:
                # Load image
                image = cv2.imread(base_image_path)
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply multiple rounds of augmentation for diversity
                for round_num in range(random.randint(2, 4)):
                    augmented = breed_augmentation(image=image)
                    image = augmented['image']
                
                # Save synthetic image
                synthetic_filename = f"{breed_name}_synthetic_{synthetic_count:04d}.jpg"
                synthetic_path = os.path.join(output_dir, synthetic_filename)
                
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(synthetic_path, image_bgr)
                
                generated_images.append(synthetic_path)
                synthetic_count += 1
                
                if synthetic_count % 100 == 0:
                    print(f"Generated {synthetic_count}/{target_count} synthetic images for {breed_name}")
                    
            except Exception as e:
                print(f"Error generating synthetic image: {e}")
                continue
        
        return generated_images
    
    def balance_dataset(self, dataset_info, min_samples_per_breed=500, output_base_dir="synthetic_data"):
        """
        Balance the entire dataset by generating synthetic images for underrepresented breeds
        """
        print("🔄 Starting dataset balancing...")
        
        balanced_dataset = {}
        generation_stats = {}
        
        for breed, images in dataset_info.items():
            current_count = len(images)
            
            if current_count < min_samples_per_breed:
                needed_samples = min_samples_per_breed - current_count
                print(f"📊 Breed '{breed}': {current_count} images, generating {needed_samples} synthetic images")
                
                breed_output_dir = os.path.join(output_base_dir, breed)
                synthetic_images = self.generate_synthetic_images(
                    images, breed, needed_samples, breed_output_dir
                )
                
                balanced_dataset[breed] = images + synthetic_images
                generation_stats[breed] = {
                    'original': current_count,
                    'synthetic': len(synthetic_images),
                    'total': len(balanced_dataset[breed])
                }
            else:
                balanced_dataset[breed] = images
                generation_stats[breed] = {
                    'original': current_count,
                    'synthetic': 0,
                    'total': current_count
                }
        
        # Print summary
        print("\n📈 Dataset Balancing Summary:")
        print("-" * 60)
        total_original = sum(stats['original'] for stats in generation_stats.values())
        total_synthetic = sum(stats['synthetic'] for stats in generation_stats.values())
        
        for breed, stats in generation_stats.items():
            print(f"{breed:20s}: {stats['original']:4d} + {stats['synthetic']:4d} = {stats['total']:4d}")
        
        print("-" * 60)
        print(f"{'TOTAL':20s}: {total_original:4d} + {total_synthetic:4d} = {total_original + total_synthetic:4d}")
        
        return balanced_dataset, generation_stats


class AdvancedCattleDataset(Dataset):
    """
    Advanced dataset class with breed-specific augmentations
    """
    
    def __init__(self, image_paths, breed_labels, gender_labels=None, age_labels=None, 
                 breed_names=None, transform=None, use_breed_specific_aug=True):
        self.image_paths = image_paths
        self.breed_labels = breed_labels
        self.gender_labels = gender_labels
        self.age_labels = age_labels
        self.breed_names = breed_names or []
        self.transform = transform
        
        # Initialize breed-specific augmentation
        if use_breed_specific_aug:
            self.breed_augmentation = CattleSpecificAugmentation()
        else:
            self.breed_augmentation = None
            
        # Base transforms for all images
        self.base_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get breed information
        breed_label = self.breed_labels[idx]
        breed_name = self.breed_names[breed_label] if breed_label < len(self.breed_names) else "Unknown"
        
        # Apply breed-specific augmentation during training
        if self.transform and self.breed_augmentation:
            breed_aug = self.breed_augmentation.get_breed_augmentation(breed_name)
            
            # Apply breed-specific augmentation with probability
            if random.random() < 0.7:  # 70% chance of breed-specific augmentation
                augmented = breed_aug(image=image)
                image = augmented['image']
        
        # Apply base transforms
        final_transform = self.base_transform(image=image)
        image = final_transform['image']
        
        # Prepare return dictionary
        sample = {
            'image': image,
            'breed': torch.tensor(breed_label, dtype=torch.long),
            'image_path': image_path,
            'breed_name': breed_name
        }
        
        # Add optional labels if available
        if self.gender_labels is not None and idx < len(self.gender_labels):
            sample['gender'] = torch.tensor(self.gender_labels[idx], dtype=torch.long)
            
        if self.age_labels is not None and idx < len(self.age_labels):
            sample['age'] = torch.tensor(self.age_labels[idx], dtype=torch.long)
        
        return sample


# Utility functions for dataset analysis and balancing
def analyze_dataset_distribution(dataset_path, breed_mapping):
    """
    Analyze the distribution of breeds in the dataset
    """
    breed_counts = defaultdict(list)
    
    for breed_folder in os.listdir(dataset_path):
        breed_path = os.path.join(dataset_path, breed_folder)
        if os.path.isdir(breed_path):
            images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images:
                breed_counts[breed_folder].append(os.path.join(breed_path, img))
    
    # Print distribution analysis
    print("📊 Dataset Distribution Analysis:")
    print("-" * 50)
    
    breed_stats = []
    for breed, images in breed_counts.items():
        count = len(images)
        breed_stats.append((breed, count))
        
    # Sort by count
    breed_stats.sort(key=lambda x: x[1], reverse=True)
    
    total_images = sum(count for _, count in breed_stats)
    
    for breed, count in breed_stats:
        percentage = (count / total_images) * 100
        print(f"{breed:25s}: {count:4d} images ({percentage:5.1f}%)")
    
    print("-" * 50)
    print(f"{'TOTAL':25s}: {total_images:4d} images")
    
    # Identify imbalanced breeds
    mean_count = total_images / len(breed_stats)
    underrepresented = [(breed, count) for breed, count in breed_stats if count < mean_count * 0.5]
    
    if underrepresented:
        print(f"\n⚠️  Underrepresented breeds (< {mean_count * 0.5:.0f} images):")
        for breed, count in underrepresented:
            print(f"  - {breed}: {count} images")
    
    return dict(breed_counts), breed_stats


if __name__ == "__main__":
    # Example usage
    print("🚀 Testing Advanced Augmentation Pipeline...")
    
    # Initialize augmentation
    aug_pipeline = CattleSpecificAugmentation()
    
    # Test breed-specific augmentation
    test_breeds = ['Gir', 'Holstein_Friesian', 'Jersey', 'Sahiwal']
    
    for breed in test_breeds:
        aug = aug_pipeline.get_breed_augmentation(breed)
        print(f"✅ {breed} augmentation pipeline initialized")
    
    # Initialize synthetic data generator
    syn_generator = SyntheticDataGenerator(aug_pipeline)
    print("✅ Synthetic data generator ready")
    
    print("\n🎯 Advanced augmentation system ready for training!")