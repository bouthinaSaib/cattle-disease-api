from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import base64
from PIL import Image
import os
import sys
import logging
import gc
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

class MultiFormatCattleDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.dataset_features = None  # Numpy array for efficient computation
        self.dataset_metadata = []     # Lightweight metadata only
        self.initialization_error = None
        self.dataset_format = None
        
        try:
            # Load the feature extraction model first
            logger.info("🚀 Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            logger.info("✅ MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading MobileNetV2 model: {e}")
            self.initialization_error = f"Model loading failed: {e}"
            return
        
        # Try to load the dataset with multi-format support
        try:
            self.load_dataset_multi_format()
            logger.info(f"✅ Dataset loaded: {len(self.dataset_metadata)} entries using {self.dataset_format} format")
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            self.initialization_error = f"Dataset loading failed: {e}"
    
    def find_dataset_file(self):
        """Find the dataset file automatically, prioritizing efficient formats"""
        # Priority order: most efficient formats first
        possible_files = [
            # NumPy format (most efficient)
            ('cattle_disease_dataset.npz', 'numpy'),
            ('cattle_disease_features.npz', 'numpy'),
            
            # HDF5 format (great for large datasets)
            ('cattle_disease_dataset.h5', 'hdf5'),
            ('cattle_disease_features.h5', 'hdf5'),
            
            # Pickle format (Python-specific but efficient)
            ('cattle_disease_dataset.pkl', 'pickle'),
            ('cattle_disease_features.pkl', 'pickle'),
            
            # Compressed JSON (fallback)
            ('cattle_disease_features_optimized.json.gz', 'json_gz'),
            ('cattle_disease_features_test_100.json.gz', 'json_gz'),
            ('cattle_disease_features.json.gz', 'json_gz'),
            
            # Regular JSON (last resort)
            ('cattle_disease_features_optimized.json', 'json'),
            ('cattle_disease_features_test_100.json', 'json'),
            ('cattle_disease_features.json', 'json')
        ]
        
        for filename, format_type in possible_files:
            if os.path.exists(filename):
                logger.info(f"📁 Found dataset file: {filename} (format: {format_type})")
                return filename, format_type
        
        return None, None
    
    def load_numpy_format(self, file_path):
        """Load NumPy .npz format"""
        logger.info("🔢 Loading NumPy format...")
        
        data = np.load(file_path)
        
        self.dataset_features = data['features'].astype(np.float32)
        classes = data['classes']
        stages = data['stages'] if 'stages' in data else [''] * len(classes)
        descriptions = data['descriptions'] if 'descriptions' in data else [''] * len(classes)
        
        # Convert numpy arrays to lists for metadata
        self.dataset_metadata = []
        for i in range(len(classes)):
            self.dataset_metadata.append({
                'class': str(classes[i]),
                'stage': str(stages[i]) if stages[i] else None,
                'description': str(descriptions[i]) if descriptions[i] else None,
                'index': i
            })
        
        logger.info(f"📊 NumPy data loaded: {self.dataset_features.shape}")
        
    def load_hdf5_format(self, file_path):
        """Load HDF5 format"""
        logger.info("📦 Loading HDF5 format...")
        
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py not available. Install with: pip install h5py")
        
        with h5py.File(file_path, 'r') as f:
            self.dataset_features = f['features'][:].astype(np.float32)
            classes = [item.decode('utf-8') for item in f['classes'][:]]
            stages = [item.decode('utf-8') for item in f['stages'][:]] if 'stages' in f else [''] * len(classes)
            descriptions = [item.decode('utf-8') for item in f['descriptions'][:]] if 'descriptions' in f else [''] * len(classes)
        
        self.dataset_metadata = []
        for i in range(len(classes)):
            self.dataset_metadata.append({
                'class': classes[i],
                'stage': stages[i] if stages[i] else None,
                'description': descriptions[i] if descriptions[i] else None,
                'index': i
            })
        
        logger.info(f"📊 HDF5 data loaded: {self.dataset_features.shape}")
        
    def load_pickle_format(self, file_path):
        """Load Pickle format"""
        logger.info("🥒 Loading Pickle format...")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_features = data['features'].astype(np.float32)
        classes = data['classes']
        stages = data.get('stages', [''] * len(classes))
        descriptions = data.get('descriptions', [''] * len(classes))
        
        self.dataset_metadata = []
        for i in range(len(classes)):
            self.dataset_metadata.append({
                'class': classes[i],
                'stage': stages[i] if stages[i] else None,
                'description': descriptions[i] if descriptions[i] else None,
                'index': i
            })
        
        logger.info(f"📊 Pickle data loaded: {self.dataset_features.shape}")
        
    def load_json_format(self, file_path, compressed=False):
        """Load JSON format (compressed or regular)"""
        if compressed:
            logger.info("🗜️  Loading compressed JSON format...")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                raw_data = json.load(f)
        else:
            logger.info("📄 Loading JSON format...")
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        # Process JSON data into efficient format
        features_list = []
        metadata_list = []
        
        for i, entry in enumerate(raw_data):
            if entry.get('features') and entry.get('class'):
                features_list.append(np.array(entry['features'], dtype=np.float32))
                metadata_list.append({
                    'class': entry.get('class', '').strip(),
                    'stage': entry.get('stage', '').strip() if entry.get('stage') else None,
                    'description': entry.get('description', '').strip() if entry.get('description') else None,
                    'index': i
                })
        
        self.dataset_features = np.array(features_list, dtype=np.float32)
        self.dataset_metadata = metadata_list
        
        logger.info(f"📊 JSON data loaded: {self.dataset_features.shape}")
        
    def load_dataset_multi_format(self):
        """Load dataset with multi-format support"""
        
        # Find the dataset file
        dataset_path, format_type = self.find_dataset_file()
        
        if not dataset_path:
            raise FileNotFoundError("No dataset file found. Supported formats: .npz, .h5, .pkl, .json.gz, .json")
        
        file_size = os.path.getsize(dataset_path)
        logger.info(f"📁 Loading dataset from: {dataset_path}")
        logger.info(f"📁 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        logger.info(f"📁 Format: {format_type}")
        
        self.dataset_format = format_type
        
        # Load based on format
        if format_type == 'numpy':
            self.load_numpy_format(dataset_path)
        elif format_type == 'hdf5':
            self.load_hdf5_format(dataset_path)
        elif format_type == 'pickle':
            self.load_pickle_format(dataset_path)
        elif format_type == 'json_gz':
            self.load_json_format(dataset_path, compressed=True)
        elif format_type == 'json':
            self.load_json_format(dataset_path, compressed=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Clean up memory
        gc.collect()
        
        logger.info(f"💾 Memory usage: {self.dataset_features.nbytes / 1024 / 1024:.1f} MB")
    
    def preprocess_image(self, image_data):
        """Preprocess image for feature extraction"""
        try:
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
            else:
                img = Image.open(io.BytesIO(image_data))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            return None
    
    def extract_features(self, img_array):
        """Extract features from preprocessed image"""
        if self.model is None:
            return None
        try:
            features = self.model.predict(img_array, verbose=0)
            return features.flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return None
    
    def find_similar_images_optimized(self, query_features, top_k=10):
        """Find similar images using optimized numpy operations"""
        if self.dataset_features is None or len(self.dataset_features) == 0:
            return []
        
        try:
            # Normalize query features
            query_norm = np.linalg.norm(query_features)
            if query_norm == 0:
                return []
            
            query_normalized = query_features / query_norm
            
            # Normalize dataset features
            dataset_norms = np.linalg.norm(self.dataset_features, axis=1)
            valid_indices = dataset_norms > 0
            
            if not np.any(valid_indices):
                return []
            
            dataset_normalized = self.dataset_features[valid_indices]
            dataset_normalized = dataset_normalized / dataset_norms[valid_indices].reshape(-1, 1)
            
            # Compute cosine similarities efficiently
            similarities = np.dot(dataset_normalized, query_normalized)
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            valid_metadata = [self.dataset_metadata[i] for i in range(len(self.dataset_metadata)) if valid_indices[i]]
            
            for idx in top_indices:
                if idx < len(valid_metadata):
                    metadata = valid_metadata[idx]
                    results.append({
                        **metadata,
                        'similarity': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error finding similar images: {e}")
            return []
    
    def classify_image(self, image_data):
        """Main classification function"""
        try:
            if self.model is None:
                return {'error': 'Model not loaded'}
            
            if self.dataset_features is None or len(self.dataset_features) == 0:
                return {'error': 'No dataset available'}
            
            # Preprocess and extract features
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return {'error': 'Failed to preprocess image'}
            
            features = self.extract_features(img_array)
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Find similar images
            similar_images = self.find_similar_images_optimized(features, top_k=10)
            
            if not similar_images:
                return {'error': 'No similar images found'}
            
            # Determine class by weighted voting
            class_votes = {}
            stage_votes = {}
            
            for item in similar_images[:5]:
                if item['class']:
                    class_votes[item['class']] = class_votes.get(item['class'], 0) + item['similarity']
                if item['stage']:
                    stage_votes[item['stage']] = stage_votes.get(item['stage'], 0) + item['similarity']
            
            best_class = max(class_votes.items(), key=lambda x: x[1])[0] if class_votes else "Unknown"
            best_stage = max(stage_votes.items(), key=lambda x: x[1])[0] if stage_votes else None
            
            confidence = np.mean([item['similarity'] for item in similar_images[:3]])
            
            return {
                'class': best_class,
                'stage': best_stage,
                'description': similar_images[0]['description'],
                'confidence': float(confidence),
                'similar_images': similar_images[:3]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in classification: {e}")
            return {'error': str(e)}

# Initialize the classifier
classifier = None
initialization_error = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    file_info = []
    try:
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                file_info.append(f"{file} ({size} bytes)")
    except:
        file_info = ["Could not list files"]
        
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None and classifier.model is not None,
        'dataset_size': len(classifier.dataset_metadata) if classifier and classifier.dataset_metadata else 0,
        'dataset_format': classifier.dataset_format if classifier else None,
        'initialization_error': classifier.initialization_error if classifier else initialization_error,
        'memory_optimized': True,
        'supported_formats': ['npz', 'h5', 'pkl', 'json.gz', 'json'],
        'files_in_directory': file_info
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        if classifier.initialization_error:
            return jsonify({'error': classifier.initialization_error}), 500
        
        # Get image data
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_data = file.read()
        elif 'image' in request.json:
            image_data = request.json['image']
        
        if image_data is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Classify
        result = classifier.classify_image(image_data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'prediction': result,
            'dataset_format': classifier.dataset_format
        })
        
    except Exception as e:
        logger.error(f"❌ Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        if classifier is None or not classifier.dataset_metadata:
            return jsonify({'error': 'No dataset loaded'}), 500
        
        # Calculate statistics
        classes = {}
        stages = {}
        
        for item in classifier.dataset_metadata:
            class_name = item['class']
            if class_name in classes:
                classes[class_name] += 1
            else:
                classes[class_name] = 1
            
            if item['stage']:
                stage_name = item['stage']
                if stage_name in stages:
                    stages[stage_name] += 1
                else:
                    stages[stage_name] = 1
        
        return jsonify({
            'success': True,
            'total_images': len(classifier.dataset_metadata),
            'total_classes': len(classes),
            'total_stages': len(stages),
            'classes': dict(sorted(classes.items())),
            'stages': dict(sorted(stages.items())),
            'dataset_format': classifier.dataset_format,
            'memory_usage_mb': classifier.dataset_features.nbytes / 1024 / 1024 if classifier.dataset_features is not None else 0
        })
        
    except Exception as e:
        logger.error(f"❌ Error in stats endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the classifier
    try:
        logger.info("🚀 Starting multi-format classifier initialization...")
        classifier = MultiFormatCattleDiseaseClassifier()
        if classifier.dataset_metadata:
            logger.info("✅ Classifier initialization completed successfully")
        else:
            logger.warning("⚠️  Classifier initialized but no dataset loaded")
    except Exception as e:
        logger.error(f"❌ Failed to initialize classifier: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        initialization_error = str(e)
        classifier = None
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
