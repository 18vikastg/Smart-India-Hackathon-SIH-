# Mobile Deployment and PWA Implementation for Cattle Recognition
import torch
import json
import os
import numpy as np
from pathlib import Path

class MobileModelConverter:
    """
    Convert PyTorch models to mobile-friendly formats (ONNX, TensorFlow Lite)
    """
    
    def __init__(self, pytorch_model, input_shape=(1, 3, 224, 224)):
        self.pytorch_model = pytorch_model
        self.input_shape = input_shape
        
    def convert_to_onnx(self, output_path="cattle_model.onnx", dynamic_batch=True):
        """
        Convert PyTorch model to ONNX format for mobile deployment
        """
        print(f"🔄 Converting model to ONNX format...")
        
        # Set model to evaluation mode
        self.pytorch_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape)
        
        # Define input/output names
        input_names = ['input_image']
        output_names = ['breed_predictions']
        
        # Handle multi-output models
        with torch.no_grad():
            sample_output = self.pytorch_model(dummy_input)
            if isinstance(sample_output, dict):
                output_names = list(sample_output.keys())
        
        # Dynamic axes for variable batch size
        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes['input_image'] = {0: 'batch_size'}
            for output_name in output_names:
                dynamic_axes[output_name] = {0: 'batch_size'}
        
        # Export to ONNX
        try:
            torch.onnx.export(
                self.pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if dynamic_batch else None
            )
            
            print(f"✅ ONNX model saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ ONNX conversion failed: {e}")
            return None
    
    def optimize_for_mobile(self, onnx_path, optimized_path=None):
        """
        Optimize ONNX model for mobile deployment
        """
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        try:
            # This would use onnxoptimizer if available
            # import onnxoptimizer
            # model = onnx.load(onnx_path)
            # optimized_model = onnxoptimizer.optimize(model)
            # onnx.save(optimized_model, optimized_path)
            
            print(f"✅ Optimized model would be saved to: {optimized_path}")
            return optimized_path
            
        except ImportError:
            print("⚠️  onnxoptimizer not available, skipping optimization")
            return onnx_path
    
    def create_model_info(self, model_path, breed_names, output_path="model_info.json"):
        """
        Create model information file for mobile app
        """
        model_info = {
            "model_path": model_path,
            "input_shape": list(self.input_shape),
            "num_classes": len(breed_names),
            "breed_names": breed_names,
            "preprocessing": {
                "resize": [224, 224],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "postprocessing": {
                "output_type": "probabilities",
                "confidence_threshold": 0.5,
                "top_k_predictions": 3
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model info saved to: {output_path}")
        return output_path


class PWAGenerator:
    """
    Generate Progressive Web App files for offline cattle recognition
    """
    
    def __init__(self, output_dir="pwa_cattle_app"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_manifest(self):
        """Generate web app manifest"""
        manifest = {
            "name": "Indian Cattle Breed Recognizer",
            "short_name": "CattleID",
            "description": "AI-powered Indian cattle breed recognition app",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#ff9933",
            "orientation": "portrait",
            "icons": [
                {
                    "src": "icons/icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "icons/icon-512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ]
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ PWA manifest created: {manifest_path}")
        return manifest_path
    
    def generate_service_worker(self):
        """Generate service worker for offline functionality"""
        
        sw_content = '''
// Service Worker for Cattle Breed Recognizer PWA
const CACHE_NAME = 'cattle-recognizer-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/style.css',
  '/app.js',
  '/tf.min.js',
  '/model.json',
  '/model_weights.bin',
  '/breed_info.json',
  '/manifest.json'
];

// Install event - cache resources
self.addEventListener('install', function(event) {
  console.log('[SW] Installing service worker');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        console.log('[SW] Caching app shell');
        return cache.addAll(urlsToCache);
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // Return cached version or fetch from network
        if (response) {
          console.log('[SW] Serving from cache:', event.request.url);
          return response;
        }
        
        console.log('[SW] Fetching from network:', event.request.url);
        return fetch(event.request);
      }
    )
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', function(event) {
  console.log('[SW] Activating service worker');
  event.waitUntil(
    caches.keys().then(function(cacheNames) {
      return Promise.all(
        cacheNames.map(function(cacheName) {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Background sync for offline predictions
self.addEventListener('sync', function(event) {
  if (event.tag === 'background-prediction') {
    console.log('[SW] Background sync triggered');
    event.waitUntil(processPendingPredictions());
  }
});

async function processPendingPredictions() {
  // Process any offline predictions when connection is restored
  const pendingPredictions = await getStoredPredictions();
  
  for (const prediction of pendingPredictions) {
    try {
      // Send to server or process locally
      await processPrediction(prediction);
      await removePendingPrediction(prediction.id);
    } catch (error) {
      console.error('[SW] Failed to process pending prediction:', error);
    }
  }
}

async function getStoredPredictions() {
  // Retrieve stored predictions from IndexedDB
  return [];  // Placeholder implementation
}

async function processPrediction(prediction) {
  // Process prediction - placeholder implementation
  console.log('[SW] Processing prediction:', prediction);
}

async function removePendingPrediction(id) {
  // Remove processed prediction from storage
  console.log('[SW] Removing pending prediction:', id);
}
'''
        
        sw_path = self.output_dir / "sw.js"
        with open(sw_path, 'w') as f:
            f.write(sw_content)
        
        print(f"✅ Service worker created: {sw_path}")
        return sw_path
    
    def generate_offline_html(self):
        """Generate offline-capable HTML page"""
        
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Cattle Breed Recognizer</title>
    <link rel="manifest" href="manifest.json">
    <meta name="theme-color" content="#ff9933">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #ff9933, #ffffff, #138808);
            min-height: 100vh;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #ff9933;
            margin: 0;
        }
        
        .upload-area {
            border: 3px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        
        .upload-area:hover {
            border-color: #ff9933;
        }
        
        .upload-area.dragover {
            border-color: #138808;
            background-color: #f0f8f0;
        }
        
        #imagePreview {
            max-width: 100%;
            max-height: 300px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .results {
            margin-top: 20px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        
        .prediction {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #ff9933;
        }
        
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #138808, #ff9933);
            transition: width 0.5s ease;
        }
        
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        
        .status.online {
            background: #d4edda;
            color: #155724;
        }
        
        .status.offline {
            background: #f8d7da;
            color: #721c24;
        }
        
        .btn {
            background: #ff9933;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        
        .btn:hover {
            background: #e6851a;
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐄 Indian Cattle Breed Recognizer</h1>
            <p>AI-powered breed identification for Indian cattle</p>
            <div id="connectionStatus" class="status offline">Offline Mode</div>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <p>📷 Click to select or drag & drop cattle image</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
        </div>
        
        <img id="imagePreview" style="display: none;">
        
        <div class="controls" style="text-align: center;">
            <button class="btn" id="predictBtn" disabled>Identify Breed</button>
            <button class="btn" id="clearBtn">Clear</button>
        </div>
        
        <div id="results" class="results">
            <h3>🎯 Prediction Results</h3>
            <div id="predictions"></div>
        </div>
    </div>
    
    <script src="tf.min.js"></script>
    <script src="app.js"></script>
</body>
</html>
'''
        
        html_path = self.output_dir / "index.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Offline HTML created: {html_path}")
        return html_path
    
    def generate_offline_js(self):
        """Generate JavaScript for offline inference"""
        
        js_content = '''
// Offline Cattle Breed Recognizer App
class OfflineCattleRecognizer {
    constructor() {
        this.model = null;
        this.breedNames = [];
        this.isModelLoaded = false;
        this.isOnline = navigator.onLine;
        
        this.initializeApp();
    }
    
    async initializeApp() {
        console.log('🚀 Initializing Cattle Recognizer App...');
        
        // Register service worker
        if ('serviceWorker' in navigator) {
            try {
                await navigator.serviceWorker.register('sw.js');
                console.log('✅ Service Worker registered');
            } catch (error) {
                console.error('❌ Service Worker registration failed:', error);
            }
        }
        
        // Load model and breed information
        await this.loadModel();
        await this.loadBreedInfo();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Update connection status
        this.updateConnectionStatus();
        
        console.log('✅ App initialized');
    }
    
    async loadModel() {
        try {
            console.log('📥 Loading TensorFlow.js model...');
            
            // Load model (assuming it's been converted to TensorFlow.js format)
            this.model = await tf.loadLayersModel('model.json');
            this.isModelLoaded = true;
            
            console.log('✅ Model loaded successfully');
            document.getElementById('predictBtn').disabled = false;
            
        } catch (error) {
            console.error('❌ Failed to load model:', error);
            this.showError('Failed to load AI model. Please check your connection.');
        }
    }
    
    async loadBreedInfo() {
        try {
            const response = await fetch('breed_info.json');
            const breedInfo = await response.json();
            this.breedNames = breedInfo.breed_names || [];
            
            console.log('✅ Breed information loaded');
        } catch (error) {
            console.error('❌ Failed to load breed info:', error);
        }
    }
    
    setupEventListeners() {
        // File input
        const imageInput = document.getElementById('imageInput');
        const uploadArea = document.getElementById('uploadArea');
        const imagePreview = document.getElementById('imagePreview');
        const predictBtn = document.getElementById('predictBtn');
        const clearBtn = document.getElementById('clearBtn');
        
        // Upload area click
        uploadArea.addEventListener('click', () => imageInput.click());
        
        // File selection
        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processImageFile(files[0]);
            }
        });
        
        // Predict button
        predictBtn.addEventListener('click', () => this.predictBreed());
        
        // Clear button
        clearBtn.addEventListener('click', () => this.clearResults());
        
        // Connection status
        window.addEventListener('online', () => this.updateConnectionStatus());
        window.addEventListener('offline', () => this.updateConnectionStatus());
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processImageFile(file);
        }
    }
    
    processImageFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            
            // Enable predict button if model is loaded
            if (this.isModelLoaded) {
                document.getElementById('predictBtn').disabled = false;
            }
        };
        reader.readAsDataURL(file);
        
        // Store image for prediction
        this.currentImageFile = file;
    }
    
    async predictBreed() {
        if (!this.isModelLoaded || !this.currentImageFile) {
            this.showError('Please load an image first.');
            return;
        }
        
        try {
            console.log('🔍 Making prediction...');
            document.getElementById('predictBtn').disabled = true;
            document.getElementById('predictBtn').textContent = 'Analyzing...';
            
            // Preprocess image
            const imageElement = document.getElementById('imagePreview');
            const tensor = await this.preprocessImage(imageElement);
            
            // Make prediction
            const predictions = await this.model.predict(tensor).data();
            
            // Process results
            const results = this.processResults(predictions);
            this.displayResults(results);
            
            // Cleanup
            tensor.dispose();
            
        } catch (error) {
            console.error('❌ Prediction failed:', error);
            this.showError('Prediction failed. Please try again.');
        } finally {
            document.getElementById('predictBtn').disabled = false;
            document.getElementById('predictBtn').textContent = 'Identify Breed';
        }
    }
    
    async preprocessImage(imageElement) {
        // Convert image to tensor and preprocess
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0);
        
        // Normalize using ImageNet stats
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        const normalized = tensor.sub(mean).div(std);
        
        // Add batch dimension
        return normalized.expandDims(0);
    }
    
    processResults(predictions) {
        // Get top 3 predictions
        const predArray = Array.from(predictions);
        const indexed = predArray.map((prob, index) => ({
            breed: this.breedNames[index] || `Breed ${index}`,
            confidence: prob,
            index: index
        }));
        
        return indexed
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
    }
    
    displayResults(results) {
        const resultsDiv = document.getElementById('results');
        const predictionsDiv = document.getElementById('predictions');
        
        predictionsDiv.innerHTML = '';
        
        results.forEach((result, index) => {
            const predictionDiv = document.createElement('div');
            predictionDiv.className = 'prediction';
            
            predictionDiv.innerHTML = `
                <h4>${result.breed}</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
            `;
            
            predictionsDiv.appendChild(predictionDiv);
        });
        
        resultsDiv.style.display = 'block';
    }
    
    clearResults() {
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('results').style.display = 'none';
        document.getElementById('imageInput').value = '';
        document.getElementById('predictBtn').disabled = true;
        this.currentImageFile = null;
    }
    
    updateConnectionStatus() {
        const statusDiv = document.getElementById('connectionStatus');
        
        if (navigator.onLine) {
            statusDiv.textContent = 'Online Mode';
            statusDiv.className = 'status online';
            this.isOnline = true;
        } else {
            statusDiv.textContent = 'Offline Mode';
            statusDiv.className = 'status offline';
            this.isOnline = false;
        }
    }
    
    showError(message) {
        // Simple error display - could be enhanced with better UI
        alert(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new OfflineCattleRecognizer();
});
'''
        
        js_path = self.output_dir / "app.js"
        with open(js_path, 'w') as f:
            f.write(js_content)
        
        print(f"✅ Offline JavaScript created: {js_path}")
        return js_path
    
    def generate_breed_info_json(self, breed_names, breed_descriptions=None):
        """Generate breed information JSON file"""
        
        breed_info = {
            "breed_names": breed_names,
            "descriptions": breed_descriptions or {},
            "total_breeds": len(breed_names)
        }
        
        breed_path = self.output_dir / "breed_info.json"
        with open(breed_path, 'w') as f:
            json.dump(breed_info, f, indent=2)
        
        print(f"✅ Breed info JSON created: {breed_path}")
        return breed_path
    
    def generate_complete_pwa(self, breed_names, breed_descriptions=None):
        """Generate complete PWA structure"""
        print(f"🏗️  Generating complete PWA in {self.output_dir}")
        
        # Create all PWA components
        manifest_path = self.generate_manifest()
        sw_path = self.generate_service_worker()
        html_path = self.generate_offline_html()
        js_path = self.generate_offline_js()
        breed_info_path = self.generate_breed_info_json(breed_names, breed_descriptions)
        
        # Create icons directory
        icons_dir = self.output_dir / "icons"
        icons_dir.mkdir(exist_ok=True)
        
        print(f"\n✅ PWA Generation Complete!")
        print(f"📁 PWA files created in: {self.output_dir}")
        print(f"📱 To deploy: Serve the directory with a web server")
        print(f"🌐 For HTTPS: Use services like Netlify, Vercel, or GitHub Pages")
        
        return {
            "manifest": manifest_path,
            "service_worker": sw_path,
            "html": html_path,
            "javascript": js_path,
            "breed_info": breed_info_path,
            "icons_dir": icons_dir
        }


# Example usage
if __name__ == "__main__":
    print("📱 Testing Mobile Deployment Components...")
    
    # Example breed names (would come from your trained model)
    indian_breeds = [
        "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi", "Hariana",
        "Ongole", "Krishna_Valley", "Nimari", "Malvi", "Kankrej", "Deoni",
        "Amritmahal", "Hallikar", "Khillari", "Bargur", "Pulikulam", "Kangayam",
        "Umblachery", "Alambadi", "Jersey", "Holstein_Friesian", "Brown_Swiss",
        "Ayrshire", "Guernsey"
    ]
    
    # Test model converter (would use actual model)
    print("🔧 Model conversion ready")
    
    # Test PWA generator
    pwa_generator = PWAGenerator("mobile_cattle_app")
    pwa_files = pwa_generator.generate_complete_pwa(indian_breeds)
    
    print("\n🎯 Mobile deployment system ready!")
    print("📋 Next steps:")
    print("  1. Convert your trained model to TensorFlow.js format")
    print("  2. Copy model files to PWA directory")
    print("  3. Test PWA locally with a web server")
    print("  4. Deploy to HTTPS hosting for full PWA features")