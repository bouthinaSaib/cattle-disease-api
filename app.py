from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging
import gc
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class LightweightCattleDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.dataset_features = None
        self.dataset_metadata = []
        self.model_loaded = False
        self.dataset_loaded = False
        self.error = None
        self.tf_available = False
        
        self.initialize()
    
    def initialize(self):
        """Initialize with lazy loading"""
        try:
            # First, load the dataset (lightweight)
            self.load_txt_dataset()
            
            # Don't load TensorFlow model immediately
            logger.info("✅ Initialization complete (model will load on first use)")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self.error = str(e)
    
    def load_model_lazy(self):
        """Load TensorFlow model only when needed"""
        if self.model_loaded:
            return True
        
        try:
            logger.info("🔄 Loading TensorFlow model (this may take a moment)...")
            
            # Import TensorFlow only when needed
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            # Store these in the object for later use
            self.tf = tf
            self.image = image  
            self.preprocess_input = preprocess_input
            
            # Load model with memory optimization
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.model_loaded = True
            self.tf_available = True
            
            logger.info("✅ TensorFlow model loaded successfully")
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading TensorFlow model: {e}")
            self.error = f"Model loading failed: {e}"
            return False
    
    def load_txt_dataset(self):
        """Load dataset from text file"""
        txt_files = [
            'cattle_disease_dataset.txt',
            'dataset.txt',
            'data.txt'
        ]
        
        dataset_file = None
        for filename in txt_files:
            if os.path.exists(filename):
                dataset_file = filename
                break
        
        if not dataset_file:
            logger.warning("⚠️  No TXT dataset file found")
            return
        
        logger.info(f"📁 Loading dataset from: {dataset_file}")
        
        features_list = []
        metadata_list = []
        
        try:
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
                        
                        # Parse features with error handling
                        try:
                            features = [float(x.strip()) for x in features_str.split(',')]
                            if len(features) == 0:
                                continue
                                
                            features_array = np.array(features, dtype=np.float32)
                            
                            # Quick validation
                            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                                continue
                                
                        except (ValueError, OverflowError):
                            continue
                        
                        features_list.append(features_array)
                        metadata_list.append({
                            'class': class_name,
                            'stage': stage,
                            'description': description,
                            'index': len(features_list) - 1
                        })
                        
                        # Limit dataset size for memory reasons
                        if len(features_list) >= 200:  # Limit to 200 entries
                            logger.info(f"📊 Limited dataset to {len(features_list)} entries for memory")
                            break
                        
                    except Exception:
                        continue
            
            if features_list:
                self.dataset_features = np.array(features_list, dtype=np.float32)
                self.dataset_metadata = metadata_list
                self.dataset_loaded = True
                
                logger.info(f"✅ Dataset loaded: {len(self.dataset_metadata)} entries")
                logger.info(f"💾 Memory: {self.dataset_features.nbytes / 1024 / 1024:.1f} MB")
                
            else:
                logger.warning("⚠️  No valid entries found")
                
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            self.error = str(e)
    
    def preprocess_image_simple(self, image_data):
        """Simple image preprocessing"""
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
            
            # Simple preprocessing (without TensorFlow preprocess_input)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            img_array = (img_array - 0.5) * 2  # Normalize to [-1, 1]
            
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing: {e}")
            return None
    
    def extract_features_with_timeout(self, img_array):
        """Extract features with memory management"""
        if not self.load_model_lazy():
            return None
            
        try:
            # Clear any previous computations
            gc.collect()
            
            # Extract features
            features = self.model.predict(img_array, verbose=0)
            result = features.flatten().astype(np.float32)
            
            # Clean up
            del features
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {e}")
            return None
    
    def find_similar_images_fast(self, query_features, top_k=5):
        """Fast similarity search with reduced k"""
        if self.dataset_features is None:
            return []
        
        try:
            # Simple cosine similarity (optimized)
            query_norm = np.linalg.norm(query_features)
            if query_norm == 0:
                return []
            
            query_normalized = query_features / query_norm
            
            # Batch normalize dataset
            dataset_norms = np.linalg.norm(self.dataset_features, axis=1)
            valid_mask = dataset_norms > 0
            
            if not np.any(valid_mask):
                return []
            
            valid_features = self.dataset_features[valid_mask]
            valid_norms = dataset_norms[valid_mask]
            
            # Compute similarities
            normalized_dataset = valid_features / valid_norms.reshape(-1, 1)
            similarities = np.dot(normalized_dataset, query_normalized)
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            valid_metadata = [self.dataset_metadata[i] for i in range(len(self.dataset_metadata)) if valid_mask[i]]
            
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
    
    def classify_image_safe(self, image_data):
        """Safe classification with error handling"""
        try:
            if not self.dataset_loaded:
                return {'error': 'Dataset not loaded'}
            
            # Preprocess image
            img_array = self.preprocess_image_simple(image_data)
            if img_array is None:
                return {'error': 'Failed to preprocess image'}
            
            # Extract features
            features = self.extract_features_with_timeout(img_array)
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Find similar images (reduced number)
            similar_images = self.find_similar_images_fast(features, top_k=5)
            
            if not similar_images:
                return {'error': 'No similar images found'}
            
            # Simple voting
            class_votes = {}
            for item in similar_images[:3]:  # Only top 3
                if item['class']:
                    class_votes[item['class']] = class_votes.get(item['class'], 0) + item['similarity']
            
            best_class = max(class_votes.items(), key=lambda x: x[1])[0] if class_votes else "Unknown"
            confidence = np.mean([item['similarity'] for item in similar_images[:2]])
            
            return {
                'class': best_class,
                'stage': similar_images[0].get('stage'),
                'description': similar_images[0].get('description'),
                'confidence': float(confidence),
                'similar_images': similar_images[:2]  # Return fewer similar images
            }
            
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return {'error': f'Classification failed: {str(e)}'}

# Initialize classifier
classifier = LightweightCattleDiseaseClassifier()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': classifier.model_loaded,
            'dataset_loaded': classifier.dataset_loaded,
            'dataset_size': len(classifier.dataset_metadata),
            'tf_available': classifier.tf_available,
            'error': classifier.error,
            'memory_info': f"{sys.getsizeof(classifier) / 1024:.1f} KB"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    """Lightweight classification endpoint"""
    try:
        # Validate request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image data
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
        result = classifier.classify_image_safe(image_data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"❌ Classify endpoint error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics"""
    try:
        if not classifier.dataset_loaded:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        classes = {}
        for item in classifier.dataset_metadata:
            class_name = item['class']
            classes[class_name] = classes.get(class_name, 0) + 1
        
        return jsonify({
            'success': True,
            'total_images': len(classifier.dataset_metadata),
            'total_classes': len(classes),
            'classes': dict(sorted(classes.items()))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
