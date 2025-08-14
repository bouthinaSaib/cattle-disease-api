from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
import json
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state - no TensorFlow needed!
app_state = {
    'dataset_loaded': False,
    'dataset_size': 0,
    'error': None,
    'features': None,
    'metadata': [],
    'feature_extractor_available': False
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
                    
                except Exception:
                    continue
        
        if features_list:
            app_state['features'] = np.array(features_list, dtype=np.float32)
            app_state['metadata'] = metadata_list
            app_state['dataset_loaded'] = True
            app_state['dataset_size'] = len(metadata_list)
            logger.info(f"✅ Dataset loaded: {len(metadata_list)} entries")
            logger.info(f"💾 Memory usage: {app_state['features'].nbytes / 1024 / 1024:.1f} MB")
        else:
            app_state['error'] = 'No valid entries found'
    
    except Exception as e:
        logger.error(f"❌ Dataset loading error: {e}")
        app_state['error'] = str(e)

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

def extract_features_from_image(image_data):
    """Extract features from image using a simple method"""
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
        
        # Convert to RGB and resize
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((64, 64), Image.Resampling.LANCZOS)  # Much smaller for speed
        img_array = np.array(img, dtype=np.float32)
        
        # Simple feature extraction (no deep learning)
        # This creates basic visual features
        features = []
        
        # Color histograms
        for channel in range(3):  # R, G, B
            hist, _ = np.histogram(img_array[:, :, channel], bins=16, range=(0, 255))
            features.extend(hist.astype(np.float32))
        
        # Texture features (simple gradients)
        gray = np.mean(img_array, axis=2)
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)
        
        # Add gradient statistics
        features.extend([
            np.mean(grad_x),
            np.std(grad_x),
            np.mean(grad_y),
            np.std(grad_y),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
        
        # Add basic image statistics
        features.extend([
            np.mean(img_array),
            np.std(img_array),
            np.min(img_array),
            np.max(img_array)
        ])
        
        # Pad or truncate to match dataset feature size
        target_size = 1280  # MobileNetV2 output size
        features_array = np.array(features, dtype=np.float32)
        
        if len(features_array) < target_size:
            # Pad with zeros
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(features_array)] = features_array
            features_array = padded
        else:
            # Truncate
            features_array = features_array[:target_size]
        
        logger.info(f"✅ Simple feature extraction successful: {len(features_array)} features")
        return features_array
        
    except Exception as e:
        logger.error(f"❌ Simple feature extraction error: {e}")
        return None

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
            'feature_extractor_available': True,  # Always available now
            'tensorflow_required': False,  # No TensorFlow needed!
            'error': app_state['error'],
            'files': files_info[:10],
            'server_working': True,
            'approach': 'simple_features'
        }
        
        return safe_json_response(health_data)
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return safe_json_response({
            'status': 'error',
            'error': str(e),
            'server_working': True
        }, 500)

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify_image():
    """Classification endpoint using simple features"""
    
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
        
        # Extract simple features (no TensorFlow needed!)
        logger.info("🔄 Extracting simple features...")
        features = extract_features_from_image(image_data)
        if features is None:
            return safe_json_response({'error': 'Failed to extract features'}, 500)
        
        # Find similar images
        logger.info("🔄 Finding similar images...")
        similar_images = find_similar_images_safe(features, top_k=5)
        if not similar_images:
            return safe_json_response({
                'error': 'No similar images found',
                'note': 'Simple feature extraction may be less accurate than deep learning'
            }, 500)
        
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
                'similar_images': similar_images[:2],
                'method': 'simple_features',
                'note': 'Using basic image features (no deep learning)'
            }
        }
        
        logger.info(f"✅ Classification complete: {best_class} (confidence: {confidence:.3f})")
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
            'classes': dict(sorted(classes.items())),
            'approach': 'simple_features'
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
    logger.info(f"🚀 Starting lightweight server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
