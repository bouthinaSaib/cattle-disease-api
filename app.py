from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
import json
import sys
import traceback
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state
app_state = {
    'dataset_loaded': False,
    'dataset_size': 0,
    'model_loaded': False,
    'model': None,
    'error': None,
    'features': None,
    'metadata': [],
    'tf_available': False
}

def safe_json_response(data, status=200):
    """Ensure we always return valid JSON"""
    try:
        json.dumps(data)
        return jsonify(data), status
    except Exception as e:
        logger.error(f"JSON serialization error: {e}")
        return jsonify({
            'error': 'Response serialization failed',
            'message': str(e)
        }), 500

def load_model_from_weights():
    """Load model by reconstructing architecture and loading weights"""
    global app_state
    
    if app_state['model_loaded']:
        return True
    
    try:
        logger.info("🔄 Starting model loading from weights...")
        
        # Look for weights file
        weights_files = [
            'mobilenetv2.weights.h5',
            'model_weights.h5',
            'weights.h5'
        ]
        
        weights_path = None
        for path in weights_files:
            if os.path.exists(path):
                weights_path = path
                logger.info(f"📁 Found weights file: {path}")
                break
        
        if not weights_path:
            available_files = [f for f in os.listdir('.') if os.path.isfile(f)]
            logger.error(f"❌ No weights file found")
            logger.error(f"❌ Available files: {available_files}")
            app_state['error'] = "No model weights found"
            return False
        
        # Check file size
        file_size = os.path.getsize(weights_path)
        logger.info(f"📊 Weights file size: {file_size / 1024 / 1024:.1f} MB")
        
        # Import TensorFlow
        logger.info("🔄 Importing TensorFlow...")
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        logger.info(f"📊 TensorFlow version: {tf.__version__}")
        
        # Create model architecture (without pre-trained weights)
        logger.info("🔄 Creating model architecture...")
        model = MobileNetV2(weights=None, include_top=False, pooling='avg')
        
        # Load the saved weights
        logger.info(f"🔄 Loading weights from: {weights_path}")
        model.load_weights(weights_path)
        
        # Store in global state
        app_state['model'] = model
        app_state['tf'] = tf
        app_state['preprocess_input'] = preprocess_input
        app_state['model_loaded'] = True
        app_state['tf_available'] = True
        
        logger.info("✅ Model loaded successfully from weights!")
        logger.info(f"📊 Model input shape: {model.input_shape}")
        logger.info(f"📊 Model output shape: {model.output_shape}")
        
        # Test the model with dummy input
        logger.info("🧪 Testing model with dummy input...")
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        logger.info(f"✅ Model test successful, output shape: {test_output.shape}")
        
        gc.collect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model from weights: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        app_state['error'] = f"Model loading failed: {e}"
        return False

def load_dataset():
    """Load dataset from text file"""
    global app_state
    
    try:
        dataset_files = ['cattle_disease_dataset.txt', 'dataset.txt', 'data.txt']
        dataset_file = None
        
        for filename in dataset_files:
            if os.path.exists(filename):
                dataset_file = filename
                break
        
        if not dataset_file:
            app_state['error'] = 'No dataset file found'
            return
        
        logger.info(f"📁 Loading dataset from: {dataset_file}")
        
        features_list = []
        metadata_list = []
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split('|', 3)
                    if len(parts) != 4:
                        continue
                    
                    class_name = parts[0].strip()
                    stage = parts[1].strip() if parts[1].strip() else None
                    description = parts[2].strip() if parts[2].strip() else None
                    features_str = parts[3].strip()
                    
                    if not class_name or not features_str:
                        continue
                    
                    features = [float(x.strip()) for x in features_str.split(',')]
                    if len(features) == 0:
                        continue
                    
                    features_array = np.array(features, dtype=np.float32)
                    
                    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                        continue
                    
                    features_list.append(features_array)
                    metadata_list.append({
                        'class': class_name,
                        'stage': stage,
                        'description': description,
                        'index': len(features_list) - 1
                    })
                    
                    # Limit dataset size for memory
                    if len(features_list) >= 200:
                        logger.info("📊 Limited dataset to 200 entries")
                        break
                    
                except Exception:
                    continue
        
        if features_list:
            app_state['features'] = np.array(features_list, dtype=np.float32)
            app_state['metadata'] = metadata_list
            app_state['dataset_loaded'] = True
            app_state['dataset_size'] = len(metadata_list)
            logger.info(f"✅ Dataset loaded: {len(metadata_list)} entries")
        else:
            app_state['error'] = 'No valid entries found'
    
    except Exception as e:
        logger.error(f"❌ Dataset loading error: {e}")
        app_state['error'] = str(e)

def preprocess_image_safe(image_data):
    """Safe image preprocessing"""
    try:
        from PIL import Image
        import io
        import base64
        
        # Decode image
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        else:
            img = Image.open(io.BytesIO(image_data))
        
        # Convert and resize
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Use TensorFlow preprocessing if available
        if app_state.get('preprocess_input'):
            img_array = app_state['preprocess_input'](img_array)
        else:
            # Fallback preprocessing for MobileNetV2
            img_array = (img_array / 127.5) - 1.0  # Scale to [-1, 1]
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing error: {e}")
        return None

def extract_features_safe(img_array):
    """Safe feature extraction with detailed error reporting"""
    try:
        logger.info("🔄 Starting feature extraction...")
        
        if not app_state['model_loaded']:
            logger.info("🔄 Model not loaded, attempting to load...")
            if not load_model_from_weights():
                logger.error("❌ Failed to load model for feature extraction")
                return None
        
        logger.info("🔄 Model is loaded, proceeding with prediction...")
        model = app_state['model']
        
        # Validate input
        logger.info(f"📊 Input shape: {img_array.shape}")
        logger.info(f"📊 Input dtype: {img_array.dtype}")
        logger.info(f"📊 Input range: [{np.min(img_array):.3f}, {np.max(img_array):.3f}]")
        
        # Predict features
        logger.info("🔄 Running model prediction...")
        features = model.predict(img_array, verbose=0)
        
        logger.info(f"📊 Features shape: {features.shape}")
        logger.info(f"📊 Features dtype: {features.dtype}")
        
        # Flatten and convert
        result = features.flatten().astype(np.float32)
        logger.info(f"📊 Final features shape: {result.shape}")
        
        # Validate output
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            logger.error("❌ Features contain NaN or Inf values")
            return None
        
        # Cleanup
        del features
        gc.collect()
        
        logger.info("✅ Feature extraction successful")
        return result
        
    except Exception as e:
        logger.error(f"❌ Feature extraction error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return None

def find_similar_images_safe(query_features, top_k=5):
    """Safe similarity search"""
    try:
        if app_state['features'] is None:
            return []
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_features)
        if query_norm == 0:
            return []
        
        query_normalized = query_features / query_norm
        
        dataset_norms = np.linalg.norm(app_state['features'], axis=1)
        valid_mask = dataset_norms > 0
        
        if not np.any(valid_mask):
            return []
        
        valid_features = app_state['features'][valid_mask]
        valid_norms = dataset_norms[valid_mask]
        
        normalized_dataset = valid_features / valid_norms.reshape(-1, 1)
        similarities = np.dot(normalized_dataset, query_normalized)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        valid_metadata = [app_state['metadata'][i] for i in range(len(app_state['metadata'])) if valid_mask[i]]
        
        for idx in top_indices:
            if idx < len(valid_metadata):
                results.append({
                    **valid_metadata[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Similarity search error: {e}")
        return []

# Load dataset on startup
load_dataset()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        files_info = []
        try:
            for file in os.listdir('.'):
                if os.path.isfile(file):
                    size = os.path.getsize(file)
                    files_info.append(f"{file} ({size} bytes)")
        except:
            files_info = ["Could not list files"]
        
        health_data = {
            'status': 'healthy',
            'dataset_loaded': app_state['dataset_loaded'],
            'dataset_size': app_state['dataset_size'],
            'model_loaded': app_state['model_loaded'],
            'tf_available': app_state['tf_available'],
            'error': app_state['error'],
            'files': files_info[:10],
            'server_working': True
        }
        
        return safe_json_response(health_data)
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return safe_json_response({
            'status': 'error',
            'error': str(e),
            'server_working': True
        }, 500)

@app.route('/test-model', methods=['POST'])
def test_model():
    """Test model loading and basic prediction"""
    try:
        # Try to load model if not loaded
        if not app_state['model_loaded']:
            logger.info("🔄 Attempting to load model for testing...")
            success = load_model_from_weights()
            if not success:
                return safe_json_response({
                    'error': 'Model loading failed',
                    'details': app_state.get('error')
                }, 500)
        
        # Create a dummy image for testing
        logger.info("🔄 Creating test image...")
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Preprocess using our function
        test_image = (test_image / 127.5) - 1.0  # MobileNetV2 preprocessing
        
        # Test model prediction
        logger.info("🔄 Testing model prediction...")
        model = app_state['model']
        features = model.predict(test_image, verbose=0)
        
        return safe_json_response({
            'success': True,
            'model_loaded': True,
            'test_features_shape': features.shape,
            'test_features_sample': features.flatten()[:5].tolist(),
            'message': 'Model test successful'
        })
        
    except Exception as e:
        logger.error(f"❌ Model test error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return safe_json_response({
            'error': f'Model test failed: {str(e)}',
            'traceback': traceback.format_exc()
        }, 500)

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify_image():
    """Classification endpoint"""
    
    if request.method == 'OPTIONS':
        return safe_json_response({'message': 'CORS preflight OK'})
    
    try:
        logger.info("🔄 Classification request received")
        
        # Check dataset
        if not app_state['dataset_loaded']:
            return safe_json_response({
                'error': 'Dataset not loaded',
                'details': app_state.get('error', 'Unknown error')
            }, 500)
        
        # Validate request
        if 'image' not in request.files and 'image' not in request.json:
            return safe_json_response({
                'error': 'No image provided'
            }, 400)
        
        # Get image data
        image_data = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return safe_json_response({'error': 'No file selected'}, 400)
            image_data = file.read()
        elif 'image' in request.json:
            image_data = request.json['image']
        
        if image_data is None:
            return safe_json_response({'error': 'Invalid image data'}, 400)
        
        # Process image
        logger.info("🔄 Preprocessing image...")
        img_array = preprocess_image_safe(image_data)
        if img_array is None:
            return safe_json_response({'error': 'Failed to preprocess image'}, 500)
        
        # Extract features
        logger.info("🔄 Extracting features...")
        features = extract_features_safe(img_array)
        if features is None:
            return safe_json_response({'error': 'Failed to extract features'}, 500)
        
        # Find similar images
        logger.info("🔄 Finding similar images...")
        similar_images = find_similar_images_safe(features, top_k=5)
        if not similar_images:
            return safe_json_response({'error': 'No similar images found'}, 500)
        
        # Classification by voting
        logger.info("🔄 Performing classification...")
        class_votes = {}
        for item in similar_images[:3]:
            if item['class']:
                class_votes[item['class']] = class_votes.get(item['class'], 0) + item['similarity']
        
        best_class = max(class_votes.items(), key=lambda x: x[1])[0] if class_votes else "Unknown"
        confidence = np.mean([item['similarity'] for item in similar_images[:2]])
        
        result = {
            'success': True,
            'prediction': {
                'class': best_class,
                'stage': similar_images[0].get('stage'),
                'description': similar_images[0].get('description'),
                'confidence': float(confidence),
                'similar_images': similar_images[:2]
            }
        }
        
        logger.info(f"✅ Classification complete: {best_class}")
        return safe_json_response(result)
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return safe_json_response({
            'error': f'Classification failed: {str(e)}'
        }, 500)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Statistics endpoint"""
    try:
        if not app_state['dataset_loaded']:
            return safe_json_response({'error': 'Dataset not loaded'}, 500)
        
        classes = {}
        for item in app_state['metadata']:
            class_name = item['class']
            classes[class_name] = classes.get(class_name, 0) + 1
        
        return safe_json_response({
            'success': True,
            'total_images': len(app_state['metadata']),
            'total_classes': len(classes),
            'classes': dict(sorted(classes.items()))
        })
        
    except Exception as e:
        return safe_json_response({'error': str(e)}, 500)

@app.errorhandler(404)
def not_found(error):
    return safe_json_response({'error': 'Endpoint not found'}, 404)

@app.errorhandler(500)
def internal_error(error):
    return safe_json_response({'error': 'Internal server error'}, 500)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
