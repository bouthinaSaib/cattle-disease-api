from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import base64
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class CattleDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.dataset_features = None
        self.dataset_metadata = []
        self.model_loaded = False
        self.dataset_loaded = False
        self.error = None
        
        self.initialize()
    
    def initialize(self):
        """Initialize model and dataset"""
        try:
            # Load TensorFlow model
            logger.info("🚀 Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.model_loaded = True
            logger.info("✅ Model loaded successfully")
            
            # Load dataset from text file
            self.load_txt_dataset()
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self.error = str(e)
    
    def load_txt_dataset(self):
        """Load dataset from text file"""
        # Look for text dataset file
        txt_files = [
            'cattle_disease_dataset.txt',
            'dataset.txt', 
            'data.txt',
            'cattle_disease_features.txt'
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
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse line: class|stage|description|features
                        parts = line.split('|', 3)  # Split into max 4 parts
                        if len(parts) != 4:
                            logger.warning(f"⚠️  Line {line_num}: Expected 4 parts, got {len(parts)}")
                            continue
                        
                        class_name = parts[0].strip()
                        stage = parts[1].strip() if parts[1].strip() else None
                        description = parts[2].strip() if parts[2].strip() else None
                        features_str = parts[3].strip()
                        
                        # Skip if no class or features
                        if not class_name or not features_str:
                            continue
                        
                        # Parse features (comma-separated numbers)
                        try:
                            features = [float(x.strip()) for x in features_str.split(',')]
                            features_array = np.array(features, dtype=np.float32)
                            
                            # Validate features
                            if len(features) == 0:
                                continue
                            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                                logger.warning(f"⚠️  Line {line_num}: Invalid features (NaN/Inf)")
                                continue
                                
                        except ValueError as e:
                            logger.warning(f"⚠️  Line {line_num}: Error parsing features - {e}")
                            continue
                        
                        # Store valid entry
                        features_list.append(features_array)
                        metadata_list.append({
                            'class': class_name,
                            'stage': stage,
                            'description': description,
                            'index': len(features_list) - 1
                        })
                        
                        # Progress logging
                        if len(features_list) % 50 == 0:
                            logger.info(f"   📊 Loaded {len(features_list)} entries...")
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Error parsing line {line_num}: {e}")
                        continue
            
            # Convert to numpy arrays
            if features_list:
                self.dataset_features = np.array(features_list, dtype=np.float32)
                self.dataset_metadata = metadata_list
                self.dataset_loaded = True
                
                logger.info(f"✅ Dataset loaded successfully:")
                logger.info(f"   📊 Total entries: {len(self.dataset_metadata)}")
                logger.info(f"   📊 Feature shape: {self.dataset_features.shape}")
                logger.info(f"   💾 Memory usage: {self.dataset_features.nbytes / 1024 / 1024:.1f} MB")
                
                # Show class distribution
                classes = {}
                for item in self.dataset_metadata:
                    class_name = item['class']
                    classes[class_name] = classes.get(class_name, 0) + 1
                
                logger.info(f"   📋 Classes found: {list(classes.keys())}")
                
            else:
                logger.warning("⚠️  No valid entries found in dataset")
                
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            self.error = str(e)
    
    def preprocess_image(self, image_data):
        """Preprocess image for feature extraction"""
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
            else:
                img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 224x224 (MobileNetV2 input size)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to array and preprocess
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
    
    def find_similar_images(self, query_features, top_k=10):
        """Find similar images using cosine similarity"""
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
            
            # Compute cosine similarities
            similarities = np.dot(dataset_normalized, query_normalized)
            
            # Get top-k results
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
            if not self.model_loaded:
                return {'error': 'Model not loaded'}
            
            if not self.dataset_loaded:
                return {'error': 'Dataset not loaded'}
            
            # Preprocess image
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return {'error': 'Failed to preprocess image'}
            
            # Extract features
            features = self.extract_features(img_array)
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Find similar images
            similar_images = self.find_similar_images(features, top_k=10)
            
            if not similar_images:
                return {'error': 'No similar images found'}
            
            # Classification by voting (top 5 similar images)
            class_votes = {}
            stage_votes = {}
            
            for item in similar_images[:5]:
                if item['class']:
                    class_votes[item['class']] = class_votes.get(item['class'], 0) + item['similarity']
                if item['stage']:
                    stage_votes[item['stage']] = stage_votes.get(item['stage'], 0) + item['similarity']
            
            # Get final prediction
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

# Initialize classifier
classifier = CattleDiseaseClassifier()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # List files in directory
        files_info = []
        for file in os.listdir('.'):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                files_info.append(f"{file} ({size} bytes)")
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': classifier.model_loaded,
            'dataset_loaded': classifier.dataset_loaded,
            'dataset_size': len(classifier.dataset_metadata),
            'feature_dimension': classifier.dataset_features.shape[1] if classifier.dataset_features is not None else 0,
            'error': classifier.error,
            'files': files_info[:10]  # Show first 10 files
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    """Image classification endpoint"""
    try:
        if not classifier.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if not classifier.dataset_loaded:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Get image data from request
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
        
        # Classify the image
        result = classifier.classify_image(image_data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"❌ Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        if not classifier.dataset_loaded:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Calculate statistics
        classes = {}
        stages = {}
        
        for item in classifier.dataset_metadata:
            class_name = item['class']
            classes[class_name] = classes.get(class_name, 0) + 1
            
            if item['stage']:
                stage_name = item['stage']
                stages[stage_name] = stages.get(stage_name, 0) + 1
        
        return jsonify({
            'success': True,
            'total_images': len(classifier.dataset_metadata),
            'total_classes': len(classes),
            'classes': dict(sorted(classes.items())),
            'stages': dict(sorted(stages.items())),
            'feature_dimension': classifier.dataset_features.shape[1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
