from flask import Flask, request, jsonify
from flask_cors import CORS
import json
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
CORS(app)  # Enable CORS for React Native

class CattleDiseaseClassifier:
    def __init__(self, dataset_path='cattle_disease_features.json'):
        # Load the feature extraction model
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        logger.info("MobileNetV2 model loaded successfully")
        
        # Load the dataset
        self.dataset = self.load_dataset(dataset_path)
        logger.info(f"Dataset loaded: {len(self.dataset)} entries")
        
    def load_dataset(self, dataset_path):
        """Load the pre-computed features dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
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
    
    def extract_features(self, img_array):
        """Extract features from preprocessed image"""
        try:
            features = self.model.predict(img_array, verbose=0)
            return features.flatten().tolist()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            return dot_product / (norm_a * norm_b)
        except:
            return 0.0
    
    def find_similar_images(self, query_features, top_k=10):
        """Find most similar images in the dataset"""
        similarities = []
        
        for item in self.dataset:
            similarity = self.cosine_similarity(query_features, item['features'])
            similarities.append({
                **item,
                'similarity': float(similarity)
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def classify_image(self, image_data):
        """Main classification function"""
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_data)
            if img_array is None:
                return None
            
            # Extract features
            features = self.extract_features(img_array)
            if features is None:
                return None
            
            # Find similar images
            similar_images = self.find_similar_images(features, top_k=10)
            
            if not similar_images:
                return None
            
            # Determine class by weighted voting (top 5 similar images)
            class_votes = {}
            stage_votes = {}
            top_similar = similar_images[:5]
            
            for item in top_similar:
                # Class voting
                if item['class']:
                    class_votes[item['class']] = class_votes.get(item['class'], 0) + item['similarity']
                
                # Stage voting - only if stage exists and is meaningful
                if item['stage'] and item['stage'].strip() and 'stade' in item['stage'].lower():
                    stage_votes[item['stage']] = stage_votes.get(item['stage'], 0) + item['similarity']
            
            # Get best class and stage
            best_class = max(class_votes.items(), key=lambda x: x[1])[0] if class_votes else "Unknown"
            
            # Only return stage if we have meaningful stage votes and good confidence
            best_stage = None
            if stage_votes:
                # Get the stage with highest vote
                potential_stage = max(stage_votes.items(), key=lambda x: x[1])[0]
                stage_confidence = stage_votes[potential_stage] / sum(stage_votes.values())
                
                # Only include stage if confidence is high enough and it's a real stage
                if stage_confidence > 0.6 and any(word in potential_stage.lower() for word in ['stade', 'stage', '1', '2', '3', '4', 'eme', 'er']):
                    best_stage = potential_stage
            
            # Get description from most similar image
            best_match = similar_images[0]
            
            # Calculate confidence (average similarity of top 3 matches)
            confidence = np.mean([item['similarity'] for item in similar_images[:3]])
            
            return {
                'class': best_class,
                'stage': best_stage,  # Will be None if no meaningful stage found
                'description': best_match['description'],
                'confidence': float(confidence),
                'similar_images': [
                    {
                        'class': item['class'],
                        'stage': item['stage'] if item['stage'] and item['stage'].strip() else None,
                        'description': item['description'],
                        'similarity': item['similarity']
                    }
                    for item in similar_images[:3]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return None

# Initialize the classifier
classifier = None

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'VetAI Cattle Disease Classification API',
        'status': 'running',
        'version': '1.0.0'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'dataset_size': len(classifier.dataset) if classifier else 0
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image data from request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_data = file.read()
        
        # Handle base64 image
        elif 'image' in request.json:
            image_data = request.json['image']
        
        if image_data is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Classify the image
        result = classifier.classify_image(image_data)
        
        if result is None:
            return jsonify({'error': 'Failed to classify image'}), 500
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available disease classes"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        classes = {}
        for item in classifier.dataset:
            class_name = item['class']
            if class_name not in classes:
                classes[class_name] = {
                    'name': class_name,
                    'count': 0,
                    'stages': set()
                }
            classes[class_name]['count'] += 1
            if item['stage']:
                classes[class_name]['stages'].add(item['stage'])
        
        # Convert sets to lists for JSON serialization
        for class_info in classes.values():
            class_info['stages'] = list(class_info['stages'])
        
        return jsonify({
            'success': True,
            'classes': list(classes.values()),
            'total_classes': len(classes),
            'total_images': len(classifier.dataset)
        })
        
    except Exception as e:
        logger.error(f"Error in classes endpoint: {e}")
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
        classifier = CattleDiseaseClassifier()
        logger.info("Classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        exit(1)
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
