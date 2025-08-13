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
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

class DiagnosticCattleDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.dataset_features = None
        self.dataset_metadata = []
        self.initialization_error = None
        self.dataset_format = None
        self.model_loaded = False
        self.dataset_loaded = False
        
        # Always initialize - don't fail if dataset is missing
        self.initialize_safely()
    
    def initialize_safely(self):
        """Initialize with comprehensive error handling"""
        
        # Step 1: Try to load the model
        try:
            logger.info("🚀 Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.model_loaded = True
            logger.info("✅ MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading MobileNetV2 model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.initialization_error = f"Model loading failed: {e}"
            return
        
        # Step 2: Try to load the dataset (but don't fail if it's missing)
        try:
            self.load_dataset_with_fallback()
        except Exception as e:
            logger.error(f"⚠️  Dataset loading failed, but continuing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't set initialization_error - we can still work without dataset for feature extraction
    
    def find_any_dataset_file(self):
        """Find any dataset file available"""
        logger.info("🔍 Looking for dataset files...")
        
        current_dir = os.getcwd()
        logger.info(f"📁 Current directory: {current_dir}")
        
        # List all files for debugging
        try:
            all_files = os.listdir('.')
            logger.info(f"📋 All files in directory:")
            for file in all_files:
                if os.path.isfile(file):
                    size = os.path.getsize(file)
                    logger.info(f"   📄 {file} ({size:,} bytes)")
        except Exception as e:
            logger.error(f"❌ Error listing files: {e}")
        
        # Try to find dataset files
        possible_files = [
            # Efficient formats
            ('cattle_disease_dataset.npz', 'numpy'),
            ('cattle_disease_features.npz', 'numpy'),
            ('cattle_disease_dataset.h5', 'hdf5'),
            ('cattle_disease_features.h5', 'hdf5'),
            ('cattle_disease_dataset.pkl', 'pickle'),
            ('cattle_disease_features.pkl', 'pickle'),
            
            # Compressed JSON
            ('cattle_disease_features_optimized.json.gz', 'json_gz'),
            ('cattle_disease_features_test_100.json.gz', 'json_gz'),
            ('cattle_disease_features.json.gz', 'json_gz'),
            
            # Regular JSON
            ('cattle_disease_features_optimized.json', 'json'),
            ('cattle_disease_features_test_100.json', 'json'),
            ('cattle_disease_features.json', 'json')
        ]
        
        for filename, format_type in possible_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                logger.info(f"✅ Found dataset file: {filename} (format: {format_type}, size: {size:,} bytes)")
                return filename, format_type
        
        logger.warning("❌ No dataset file found")
        return None, None
    
    def load_dataset_with_fallback(self):
        """Load dataset with comprehensive fallback"""
        
        dataset_path, format_type = self.find_any_dataset_file()
        
        if not dataset_path:
            logger.warning("⚠️  No dataset file found - running in feature extraction only mode")
            return
        
        file_size = os.path.getsize(dataset_path)
        logger.info(f"📁 Loading dataset: {dataset_path} ({file_size:,} bytes, format: {format_type})")
        
        self.dataset_format = format_type
        
        try:
            if format_type == 'json_gz':
                self.load_json_format(dataset_path, compressed=True)
            elif format_type == 'json':
                self.load_json_format(dataset_path, compressed=False)
            elif format_type == 'numpy':
                self.load_numpy_format(dataset_path)
            elif format_type == 'hdf5':
                self.load_hdf5_format(dataset_path)
            elif format_type == 'pickle':
                self.load_pickle_format(dataset_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.dataset_loaded = True
            logger.info(f"✅ Dataset loaded successfully: {len(self.dataset_metadata)} entries")
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def load_json_format(self, file_path, compressed=False):
        """Load JSON format with error handling"""
        try:
            if compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    raw_data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            
            logger.info(f"📊 JSON data loaded: {len(raw_data)} entries")
            
            # Process into efficient format
            features_list = []
            metadata_list = []
            
            for i, entry in enumerate(raw_data[:100]):  # Limit to first 100 for safety
                if entry.get('features') and entry.get('class'):
                    try:
                        features = np.array(entry['features'], dtype=np.float32)
                        if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                            features_list.append(features)
                            metadata_list.append({
                                'class': str(entry.get('class', '')).strip(),
                                'stage': str(entry.get('stage', '')).strip() if entry.get('stage') else None,
                                'description': str(entry.get('description', '')).strip() if entry.get('description') else None,
                                'index': len(features_list) - 1
                            })
                    except Exception as e:
                        logger.warning(f"⚠️  Skipping entry {i}: {e}")
                        continue
            
            if features_list:
                self.dataset_features = np.array(features_list, dtype=np.float32)
                self.dataset_metadata = metadata_list
                logger.info(f"📊 Processed dataset: {self.dataset_features.shape}")
            else:
                raise ValueError("No valid entries found in dataset")
                
        except Exception as e:
            logger.error(f"❌ JSON loading error: {e}")
            raise
    
    def load_numpy_format(self, file_path):
        """Load NumPy format"""
        try:
            data = np.load(file_path)
            self.dataset_features = data['features'].astype(np.float32)
            
            classes = data['classes']
            stages = data.get('stages', [''] * len(classes))
            descriptions = data.get('descriptions', [''] * len(classes))
            
            self.dataset_metadata = []
            for i in range(len(classes)):
                self.dataset_metadata.append({
                    'class': str(classes[i]),
                    'stage': str(stages[i]) if stages[i] else None,
                    'description': str(descriptions[i]) if descriptions[i] else None,
                    'index': i
                })
                
            logger.info(f"📊 NumPy data loaded: {self.dataset_features.shape}")
        except Exception as e:
            logger.error(f"❌ NumPy loading error: {e}")
            raise
    
    def load_hdf5_format(self, file_path):
        """Load HDF5 format"""
        try:
            import h5py
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
        except ImportError:
            raise ImportError("h5py not available for HDF5 format")
        except Exception as e:
            logger.error(f"❌ HDF5 loading error: {e}")
            raise
    
    def load_pickle_format(self, file_path):
        """Load Pickle format"""
        try:
            import pickle
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
        except Exception as e:
            logger.error(f"❌ Pickle loading error: {e}")
            raise
    
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
        if not self.model_loaded:
            return None
        try:
            features = self.model.predict(img_array, verbose=0)
            return features.flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return None

# Initialize the classifier
classifier = None
initialization_error = None

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    
    file_info = []
    try:
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                file_info.append(f"{file} ({size} bytes)")
    except:
        file_info = ["Could not list files"]
    
    health_data = {
        'status': 'healthy',
        'classifier_exists': classifier is not None,
        'model_loaded': classifier.model_loaded if classifier else False,
        'dataset_loaded': classifier.dataset_loaded if classifier else False,
        'dataset_size': len(classifier.dataset_metadata) if classifier and classifier.dataset_metadata else 0,
        'dataset_format': classifier.dataset_format if classifier else None,
        'initialization_error': classifier.initialization_error if classifier else initialization_error,
        'files_in_directory': file_info,
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'current_directory': os.getcwd()
    }
    
    return jsonify(health_data)

@app.route('/extract-features', methods=['POST'])
def extract_features_only():
    """Extract features from image (works even without dataset)"""
    try:
        if classifier is None or not classifier.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
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
        
        # Preprocess and extract features
        img_array = classifier.preprocess_image(image_data)
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        features = classifier.extract_features(img_array)
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        return jsonify({
            'success': True,
            'features_length': len(features),
            'features_sample': features[:10].tolist(),
            'message': 'Feature extraction successful',
            'note': 'Classification requires dataset to be loaded'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in extract-features: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classification endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        if not classifier.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if not classifier.dataset_loaded:
            return jsonify({'error': 'Dataset not loaded - only feature extraction available'}), 500
        
        return jsonify({'error': 'Classification endpoint under construction'}), 503
        
    except Exception as e:
        logger.error(f"❌ Error in classify: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize with comprehensive error handling
    try:
        logger.info("🚀 Starting diagnostic classifier...")
        classifier = DiagnosticCattleDiseaseClassifier()
        logger.info("✅ Diagnostic classifier ready")
    except Exception as e:
        logger.error(f"❌ Critical initialization error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        initialization_error = str(e)
        classifier = None
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
