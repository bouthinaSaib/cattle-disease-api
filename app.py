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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

class CattleDiseaseClassifier:
    def __init__(self, dataset_path='cattle_disease_features.json'):
        self.model = None
        self.dataset = []
        self.initialization_error = None
        
        try:
            # Load the feature extraction model first
            logger.info("Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2 model: {e}")
            self.initialization_error = f"Model loading failed: {e}"
            return
        
        # Try to load the dataset
        try:
            self.dataset = self.load_dataset(dataset_path)
            logger.info(f"Dataset loaded: {len(self.dataset)} entries")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.initialization_error = f"Dataset loading failed: {e}"
            # Continue without dataset for debugging
        
    def load_dataset(self, dataset_path):
        """Load the pre-computed features dataset with extensive error handling"""
        
        # Get the absolute path
        if not os.path.isabs(dataset_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(script_dir, dataset_path)
        
        logger.info(f"Looking for dataset at: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            # Try compressed version
            compressed_path = dataset_path + '.gz'
            if os.path.exists(compressed_path):
                dataset_path = compressed_path
                logger.info(f"Using compressed version: {compressed_path}")
            else:
                logger.error(f"Dataset file not found at: {dataset_path}")
                return []
        
        file_size = os.path.getsize(dataset_path)
        logger.info(f"Dataset file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_size == 0:
            logger.error("Dataset file is empty!")
            return []
        
        if file_size > 100 * 1024 * 1024:  # > 100MB
            logger.warning("Dataset file is very large, this might cause memory issues")
        
        try:
            # Read first few bytes to check format
            with open(dataset_path, 'rb') as f:
                first_bytes = f.read(10)
                logger.info(f"File starts with: {first_bytes}")
            
            # Determine if compressed
            is_compressed = dataset_path.endswith('.gz') or first_bytes.startswith(b'\x1f\x8b')
            
            if is_compressed:
                logger.info("Loading compressed dataset...")
                with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
                    # Load in smaller chunks for large files
                    content = f.read()
            else:
                logger.info("Loading regular dataset...")
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            logger.info(f"File content loaded, size: {len(content)} characters")
            
            # Parse JSON
            logger.info("Parsing JSON...")
            dataset = json.loads(content)
            
            logger.info(f"JSON parsed successfully, type: {type(dataset)}")
            
            if isinstance(dataset, list):
                logger.info(f"Dataset is a list with {len(dataset)} items")
                if len(dataset) > 0:
                    first_item = dataset[0]
                    logger.info(f"First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        logger.info(f"First item keys: {list(first_item.keys())}")
                        
                        # Validate structure
                        required_keys = ['features', 'class']
                        for key in required_keys:
                            if key not in first_item:
                                logger.warning(f"Missing required key: {key}")
                            else:
                                logger.info(f"✓ Found required key: {key}")
                
                return dataset
            else:
                logger.error(f"Expected list, got {type(dataset)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
            
            # Show problematic content
            try:
                lines = content.split('\n')
                if e.lineno <= len(lines):
                    logger.error(f"Problematic line: {lines[e.lineno - 1][:200]}...")
            except:
                pass
            return []
            
        except MemoryError as e:
            logger.error(f"Memory error loading dataset: {e}")
            logger.error("File is too large for available memory")
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error loading dataset: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def extract_features(self, img_array):
        """Extract features from preprocessed image"""
        if self.model is None:
            logger.error("Model not loaded")
            return None
            
        try:
            features = self.model.predict(img_array, verbose=0)
            return features.flatten().tolist()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def preprocess_image(self, image_data):
        """Preprocess image for feature extraction"""
        try:
            # If image_data is base64 string, decode it
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
            else:
                img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Add batch dimension and preprocess
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

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
        'dataset_size': len(classifier.dataset) if classifier and classifier.dataset else 0,
        'initialization_error': classifier.initialization_error if classifier else initialization_error,
        'current_directory': os.getcwd(),
        'files_in_directory': file_info
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint"""
    debug_data = {
        'classifier_exists': classifier is not None,
        'model_loaded': classifier is not None and classifier.model is not None,
        'dataset_loaded': classifier is not None and len(classifier.dataset) > 0,
        'dataset_size': len(classifier.dataset) if classifier and classifier.dataset else 0,
        'initialization_error': classifier.initialization_error if classifier else initialization_error,
        'current_directory': os.getcwd(),
        'python_version': sys.version
    }
    
    # File information
    files_info = []
    try:
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                files_info.append({
                    'name': file,
                    'size_bytes': size,
                    'size_mb': round(size / 1024 / 1024, 2)
                })
    except Exception as e:
        files_info = [f"Error listing files: {e}"]
    
    debug_data['files'] = files_info
    
    return jsonify(debug_data)

@app.route('/extract-features', methods=['POST'])
def extract_features_only():
    """Extract features from image without classification (for testing)"""
    try:
        if classifier is None or classifier.model is None:
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
        
        # Preprocess image
        img_array = classifier.preprocess_image(image_data)
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
        
        # Extract features
        features = classifier.extract_features(img_array)
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        return jsonify({
            'success': True,
            'features_length': len(features),
            'features_sample': features[:10],  # First 10 features
            'message': 'Feature extraction successful'
        })
        
    except Exception as e:
        logger.error(f"Error in extract-features endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
            
        if classifier.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        if not classifier.dataset:
            return jsonify({'error': 'No dataset available - check /debug for details'}), 500
        
        # Get image data from request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        return jsonify({'error': 'Classification temporarily disabled for debugging'}), 503
        
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the classifier
    try:
        logger.info("Starting classifier initialization...")
        classifier = CattleDiseaseClassifier()
        logger.info("Classifier initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        initialization_error = str(e)
        classifier = None
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
