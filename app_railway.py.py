#!/usr/bin/env python3
"""
Production Toxicity Detection API (Railway Deployment Version)
Compatible with Python 3.11 - Uses traditional ML models
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging
import os
import json
import re
import joblib
from datetime import datetime

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ToxicityDetector:
    def __init__(self, model_path='toxicity_sklearn.pkl', vectorizer_path='vectorizer_sklearn.pkl', config_path='model_config_sklearn.json'):
        """
        Initialize the toxicity detector with scikit-learn model
        """
        self.model = None
        self.vectorizer = None
        self.config = None
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Try to load NLTK stopwords, fallback to basic list
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            self.stop_words = set(stopwords.words('english'))
            self.word_tokenize = word_tokenize
            self.use_nltk = True
            logger.info("NLTK loaded successfully")
        except:
            logger.warning("NLTK not available, using basic preprocessing")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.use_nltk = False
        
        # Load model components
        self.load_model_components(model_path, vectorizer_path, config_path)
        
        if self.is_loaded():
            logger.info("ToxicityDetector (Scikit-Learn) initialized successfully with trained model")
        else:
            logger.warning("ToxicityDetector initialized in mock mode - model files not found")
    
    def load_model_components(self, model_path, vectorizer_path, config_path):
        """Load model, vectorizer, and configuration"""
        try:
            # Load configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Model configuration loaded")
            else:
                # Default configuration
                self.config = {
                    'max_features': 50000,
                    'ngram_range': [1, 2],
                    'target_columns': self.categories,
                    'model_type': 'scikit-learn'
                }
                logger.warning("Using default configuration")
            
            # Load model
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
                self.model_loaded = True
            else:
                logger.error(f"Model file not found: {model_path}")
                self.model_loaded = False
            
            # Load vectorizer
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Vectorizer loaded from {vectorizer_path}")
                self.vectorizer_loaded = True
            else:
                logger.error(f"Vectorizer file not found: {vectorizer_path}")
                self.vectorizer_loaded = False
            
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            self.model_loaded = False
            self.vectorizer_loaded = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and remove stopwords
        if self.use_nltk:
            try:
                tokens = self.word_tokenize(text)
                tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
                text = ' '.join(tokens)
            except:
                # Fallback to basic processing
                words = text.split()
                words = [word for word in words if word not in self.stop_words and len(word) > 2]
                text = ' '.join(words)
        else:
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            text = ' '.join(words)
        
        return text.strip()
    
    def _mock_predict(self, text):
        """Generate mock predictions for testing when model is not available"""
        text_lower = text.lower()
        
        # Define some basic toxic keywords for each category
        toxic_keywords = {
            'toxic': ['hate', 'stupid', 'idiot', 'kill', 'die', 'terrible', 'awful'],
            'severe_toxic': ['kill yourself', 'die', 'murder', 'extreme'],
            'obscene': ['fuck', 'shit', 'damn', 'ass', 'hell'],
            'threat': ['kill', 'murder', 'hurt', 'destroy', 'attack'],
            'insult': ['stupid', 'idiot', 'moron', 'loser', 'dumb'],
            'identity_hate': ['racist', 'sexist', 'homophobic', 'discrimination']
        }
        
        scores = {}
        for category in self.categories:
            # Check for keywords
            keyword_count = sum(1 for keyword in toxic_keywords.get(category, []) 
                              if keyword in text_lower)
            
            # Base score on keyword presence and text characteristics
            base_score = 0.05  # Very low base score
            
            if keyword_count > 0:
                base_score = 0.3 + (keyword_count * 0.2)  # Increase based on keyword count
            
            # Add some factors based on text characteristics
            text_factor = min(len(text) / 1000, 0.1)
            caps_factor = sum(1 for c in text if c.isupper()) / max(len(text), 1) * 0.1
            exclamation_factor = text.count('!') * 0.05
            
            final_score = base_score + text_factor + caps_factor + exclamation_factor
            scores[category] = min(final_score, 0.9)  # Cap at 90%
        
        logger.info(f"Mock prediction generated for text: '{text[:50]}...'")
        return scores
    
    def predict(self, text):
        """Predict toxicity scores for given text"""
        try:
            if not self.model_loaded or not self.vectorizer_loaded:
                logger.warning("Model or vectorizer not loaded, using mock predictions")
                return self._mock_predict(text)
            
            if not text or len(text.strip()) == 0:
                return {category: 0.01 for category in self.categories}
            
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            if not clean_text:
                return {category: 0.01 for category in self.categories}
            
            # Vectorize text
            text_vector = self.vectorizer.transform([clean_text])
            
            # Make prediction
            if hasattr(self.model, 'estimators_'):
                predictions = []
                for i, category in enumerate(self.categories):
                    estimator = self.model.estimators_[i]
                    if hasattr(estimator, 'predict_proba'):
                        proba = estimator.predict_proba(text_vector)
                        if proba.shape[1] == 2:
                            predictions.append(proba[0, 1])
                        else:
                            predictions.append(proba[0, 0])
                    else:
                        pred = estimator.predict(text_vector)
                        predictions.append(float(pred[0]))
                
                scores = {category: float(pred) for category, pred in zip(self.categories, predictions)}
            
            elif hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(text_vector)
                scores = {category: float(proba[0, i]) for i, category in enumerate(self.categories)}
            
            else:
                predictions = self.model.predict(text_vector)
                if len(predictions.shape) == 2 and predictions.shape[1] == len(self.categories):
                    scores = {category: float(predictions[0, i]) for i, category in enumerate(self.categories)}
                else:
                    return self._mock_predict(text)
            
            # Ensure scores are in valid range [0, 1]
            for category in scores:
                scores[category] = max(0.0, min(1.0, scores[category]))
            
            return scores
                
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            return self._mock_predict(text)
    
    def is_loaded(self):
        """Check if model components are properly loaded"""
        return hasattr(self, 'model_loaded') and hasattr(self, 'vectorizer_loaded') and \
               self.model_loaded and self.vectorizer_loaded

# Initialize the detector
detector = ToxicityDetector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        model_status = "🤖 Scikit-Learn Model Loaded" if detector.is_loaded() else "⚠️ Mock Mode (Model Not Found)"
        description = "Traditional machine learning toxicity detection is active." if detector.is_loaded() else "Using simple keyword matching for testing."
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Toxicity Detection API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; max-width: 800px; }}
                h1 {{ color: #333; }}
                .status {{ padding: 10px; border-radius: 5px; margin: 20px 0; }}
                .loaded {{ background: #d4edda; color: #155724; }}
                .mock {{ background: #fff3cd; color: #856404; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .endpoint {{ margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛡️ Toxicity Detection API</h1>
                <div class="status {'loaded' if detector.is_loaded() else 'mock'}">
                    <strong>Status:</strong> {model_status}<br>
                    {description}
                </div>
                
                <h2>🔗 API Endpoints</h2>
                <div class="endpoint"><strong>POST</strong> /analyze - Analyze text for toxicity</div>
                <div class="endpoint"><strong>GET</strong> /health - Health check</div>
                <div class="endpoint"><strong>GET</strong> /api/info - API information</div>
                <div class="endpoint"><strong>POST</strong> /batch_analyze - Batch analysis</div>
                
                <h3>📝 Example Usage</h3>
                <pre>curl -X POST -H "Content-Type: application/json" \\
  -d '{{"text":"Hello world"}}' \\
  https://your-app.railway.app/analyze</pre>
                
                <h3>📱 Test the API</h3>
                <textarea id="testText" placeholder="Enter text to test..." style="width: 100%; height: 100px; margin: 10px 0; padding: 10px;"></textarea>
                <button onclick="testAPI()" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">Test API</button>
                <div id="result" style="margin-top: 20px;"></div>
                
                <script>
                async function testAPI() {{
                    const text = document.getElementById('testText').value;
                    const resultDiv = document.getElementById('result');
                    
                    if (!text.trim()) {{
                        resultDiv.innerHTML = '<div style="color: red;">Please enter some text to test.</div>';
                        return;
                    }}
                    
                    try {{
                        resultDiv.innerHTML = '<div style="color: blue;">Testing...</div>';
                        const response = await fetch('/analyze', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{text: text}})
                        }});
                        
                        const data = await response.json();
                        resultDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }} catch (error) {{
                        resultDiv.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
                    }}
                }}
                </script>
            </div>
        </body>
        </html>
        """

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for toxicity"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 2000:
            return jsonify({'error': 'Text too long (max 2000 characters)'}), 400
        
        logger.info(f"Analyzing text: '{text[:100]}...'")
        
        # Make prediction
        scores = detector.predict(text)
        
        # Calculate overall risk level
        max_score = max(scores.values())
        if max_score < 0.3:
            risk_level = 'low'
        elif max_score < 0.7:
            risk_level = 'moderate'
        else:
            risk_level = 'high'
        
        response = {
            'success': True,
            'scores': scores,
            'risk_level': risk_level,
            'max_score': max_score,
            'text_length': len(text),
            'model_type': 'scikit_learn' if detector.is_loaded() else 'mock',
            'message': 'Prediction from scikit-learn model' if detector.is_loaded() else 'Mock prediction for testing'
        }
        
        logger.info(f"Analysis completed - Risk: {risk_level}, Max score: {max_score:.3f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway"""
    model_status = detector.is_loaded()
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'mode': 'production' if model_status else 'mock',
        'message': 'Toxicity detection API is running on Railway',
        'model_type': 'Scikit-Learn' if model_status else 'Mock',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information"""
    model_status = detector.is_loaded()
    
    info = {
        'name': 'Toxicity Detection API',
        'version': '2.0.0-railway',
        'mode': 'production' if model_status else 'mock',
        'platform': 'Railway',
        'description': 'Traditional ML toxicity detection API' if model_status else 'Mock toxicity detection for testing',
        'categories': detector.categories,
        'max_text_length': 2000,
        'model_type': 'TF-IDF + Multi-Output Classifier' if model_status else 'Rule-based keyword matching',
        'endpoints': {
            '/': 'Frontend interface',
            '/analyze': 'POST - Analyze text for toxicity',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information',
            '/batch_analyze': 'POST - Batch analysis'
        }
    }
    
    if model_status and detector.config:
        info['model_config'] = detector.config
    
    return jsonify(info)

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts in batch"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list'}), 400
        
        if len(texts) > 50:  # Reduced for Railway limits
            return jsonify({'error': 'Maximum 50 texts per batch'}), 400
        
        results = []
        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                continue
                
            if len(text) > 2000:
                results.append({
                    'index': i,
                    'error': 'Text too long (max 2000 characters)'
                })
                continue
            
            try:
                scores = detector.predict(text.strip())
                max_score = max(scores.values())
                
                if max_score < 0.3:
                    risk_level = 'low'
                elif max_score < 0.7:
                    risk_level = 'moderate'
                else:
                    risk_level = 'high'
                
                results.append({
                    'index': i,
                    'scores': scores,
                    'risk_level': risk_level,
                    'max_score': max_score,
                    'text_length': len(text)
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': f'Analysis failed: {str(e)}'
                })
        
        response = {
            'success': True,
            'results': results,
            'total_processed': len(results),
            'model_type': 'scikit_learn' if detector.is_loaded() else 'mock'
        }
        
        logger.info(f"Batch analysis completed for {len(texts)} texts")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print("🚀 Starting Toxicity Detection API on Railway...")
    
    if detector.is_loaded():
        print("🤖 AI Model Status: LOADED")
        print("📊 Model Type: Traditional Machine Learning (Scikit-Learn)")
        print("🎯 Categories: " + ", ".join(detector.categories))
    else:
        print("⚠️  AI Model Status: NOT LOADED (Using Mock Mode)")
        print("🔧 The app will work with keyword-based predictions")
    
    print(f"🌐 Server starting on {host}:{port}")
    
    # Run the Flask app
    app.run(debug=False, host=host, port=port)