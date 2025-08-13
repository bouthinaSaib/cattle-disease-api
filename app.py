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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

class CattleDiseaseClassifier:
    def __init__(self, dataset_path='cattle_disease_features.json'):
        # Get the absolute path to the dataset file
        if not os.path.isabs(dataset_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(script_dir, dataset_path)
        
        # Also check for compressed version
        compressed_path = dataset_path + '.gz'
        
        logger.info(f"Looking for dataset at: {dataset_path}")
        logger.info(f"Looking for compressed dataset at: {compressed_path}")
        logger.info(f"Regular file exists: {os.path.exists(dataset_path)}")
        logger.info(f"Compressed file exists: {os.path.exists(compressed_path)}")
        
        if os.path.exists(dataset_path):
            logger.info(f"Using regular file, size: {os.path.getsize(dataset_path)} bytes")
        elif os.path.exists(compressed_path):
            logger.info(f"Using compressed file, size: {os.path.getsize(compressed_path)} bytes")
            dataset_path = compressed_path
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Load the feature extraction model
        try:
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MobileNetV2 model: {e}")
            raise
        
        # Load the dataset
        self.dataset = self.load_dataset(dataset_path)
        logger.info(f"Dataset loaded: {len(self.dataset)} entries")
        
    def load_dataset(self, dataset_path):
        """Load the pre-computed features dataset (supports both regular and gzipped files)"""
        try:
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset file not found at: {dataset_path}")
                logger.info("Available files in directory:")
                try:
                    for file in os.listdir(os.path.dirname(dataset_path) if os.path.dirname(dataset_path) else '.'):
                        file_path = os.path.join(os.path.dirname(dataset_path) if os.path.dirname(dataset_path) else '.', file)
                        size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                        logger.info(f"  - {file} ({size} bytes)")
                except Exception as e:
                    logger.info(f"  Could not list directory contents: {e}")
                return []
            
            # Get file size for debugging
            file_size = os.path.getsize(dataset_path)
            logger.info(f"Dataset file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # Check if file is empty
            if file_size == 0:
                logger.error("Dataset file is empty!")
                return []
            
            # Try to read first few bytes to check file format
            try:
                with open(dataset_path, 'rb') as f:
                    first_bytes = f.read(100)
                    logger.info(f"First 100 bytes: {first_bytes}")
            except Exception as e:
                logger.error(f"Could not read first bytes: {e}")
            
            # Check if file is gzipped
            if dataset_path.endswith('.gz'):
                logger.info("Loading compressed dataset...")
                with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                logger.info("Loading regular dataset...")
                try:
                    # For large files, try to optimize memory usage
                    if file_size > 50 * 1024 * 1024:  # > 50MB
                        logger.info("Large file detected, using optimized loading...")
                        import gc
                        gc.collect()  # Clear memory before loading
                    
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        logger.info("File opened successfully, attempting to parse JSON...")
                        
                        # Read in chunks to detect encoding issues early
                        test_chunk = f.read(1024)
                        f.seek(0)  # Reset to beginning
                        
                        logger.info(f"Test chunk read successfully: {len(test_chunk)} chars")
                        
                        dataset = json.load(f)
                        logger.info("JSON parsed successfully")
                        
                except UnicodeDecodeError as e:
                    logger.error(f"Unicode decode error: {e}")
                    logger.info("Trying with different encoding...")
                    try:
                        with open(dataset_path, 'r', encoding='latin-1') as f:
                            dataset = json.load(f)
                    except Exception as e2:
                        logger.error(f"Failed with latin-1 encoding: {e2}")
                        logger.info("Trying with utf-8-sig (BOM handling)...")
                        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
                            dataset = json.load(f)
                except MemoryError as e:
                    logger.error(f"Memory error - file too large: {e}")
                    logger.info("Try compressing the file or reducing dataset size")
                    return []
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    # Try to fix common issues
                    logger.info("Attempting to fix common JSON issues...")
                    try:
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Remove BOM if present
                        if content.startswith('\ufeff'):
                            content = content[1:]
                            logger.info("Removed BOM from file")
                        
                        # Try parsing the cleaned content
                        dataset = json.loads(content)
                        logger.info("Successfully parsed after cleaning")
                    except Exception as fix_error:
                        logger.error(f"Could not fix JSON: {fix_error}")
                        raise e
                    
            logger.info(f"Successfully loaded dataset with {len(dataset)} entries")
            
            # Validate dataset structure
            if len(dataset) > 0:
                first_item = dataset[0]
                logger.info(f"First item keys: {list(first_item.keys())}")
                required_keys = ['features', 'class']
                missing_keys = [key for key in required_keys if key not in first_item]
                if missing_keys:
                    logger.warning(f"Missing required keys in dataset: {missing_keys}")
                else:
                    logger.info("Dataset structure validation passed")
            
            return dataset
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
            # Try to read a small sample to see what's wrong
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1000)
                    logger.info(f"First 1000 characters: {sample}")
            except:
                pass
            return []
        except MemoryError as e:
            logger.error(f"Memory error - file too large to load: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading dataset: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            if not self.dataset:
                logger.error("No dataset available for classification")
                return None
                
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
                
                # Stage voting
                if item['stage']:
                    stage_votes[item['stage']] = stage_votes.get(item['stage'], 0) + item['similarity']
            
            # Get best class and stage
            best_class = max(class_votes.items(), key=lambda x: x[1])[0] if class_votes else "Unknown"
            best_stage = max(stage_votes.items(), key=lambda x: x[1])[0] if stage_votes else None
            
            # Get description from most similar image
            best_match = similar_images[0]
            
            # Calculate confidence (average similarity of top 3 matches)
            confidence = np.mean([item['similarity'] for item in similar_images[:3]])
            
            return {
                'class': best_class,
                'stage': best_stage,
                'description': best_match['description'],
                'confidence': float(confidence),
                'similar_images': [
                    {
                        'class': item['class'],
                        'stage': item['stage'],
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
        'dataset_size': len(classifier.dataset) if classifier else 0,
        'current_directory': os.getcwd(),
        'files_in_directory': file_info
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        if not classifier.dataset:
            return jsonify({'error': 'No dataset available'}), 500
        
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
            
        if not classifier.dataset:
            return jsonify({'error': 'No dataset available'}), 500
        
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
        if classifier.dataset:
            logger.info("Classifier initialized successfully")
        else:
            logger.warning("Classifier initialized but no dataset loaded")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        classifier = None
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
