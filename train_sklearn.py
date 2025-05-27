#!/usr/bin/env python3
"""
Toxicity Detection Model Training Script (Scikit-Learn Version)
Compatible with Python 3.13 - Uses traditional ML instead of deep learning
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import re
import string
import logging
import joblib
from datetime import datetime
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_sklearn.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ToxicityModelTrainer:
    def __init__(self, max_features=50000, ngram_range=(1, 2)):
        """
        Initialize the toxicity model trainer (Scikit-Learn version)
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
        # Target columns for toxicity detection
        self.target_columns = [
            'toxic', 'severe_toxic', 'obscene', 
            'threat', 'insult', 'identity_hate'
        ]
        
        logger.info("ToxicityModelTrainer (Scikit-Learn) initialized")
        logger.info(f"Max features: {max_features}")
        logger.info(f"N-gram range: {ngram_range}")
    
    def load_data(self, filepath='train.csv'):
        """Load and validate the training data"""
        try:
            logger.info(f"Loading data from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Training data file '{filepath}' not found!")
            
            # Load data
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['comment_text'] + self.target_columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Basic data info
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Total comments: {len(df)}")
            logger.info(f"Null values in comment_text: {df['comment_text'].isnull().sum()}")
            
            # Toxicity statistics
            for col in self.target_columns:
                toxic_count = df[col].sum()
                toxic_pct = (toxic_count / len(df)) * 100
                logger.info(f"{col}: {toxic_count} ({toxic_pct:.2f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and processed text
        """
        if pd.isna(text):
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
        try:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            text = ' '.join(tokens)
        except:
            # Fallback if NLTK fails
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            text = ' '.join(words)
        
        return text.strip()
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: Processed training and testing data
        """
        logger.info("Preparing data for training...")
        
        # Remove rows with missing comment_text
        df = df.dropna(subset=['comment_text'])
        logger.info(f"Data shape after removing nulls: {df.shape}")
        
        # Take a sample for faster training (optional)
        if len(df) > 100000:
            logger.info("Sampling data for faster training...")
            df = df.sample(n=100000, random_state=random_state)
            logger.info(f"Sampled data shape: {df.shape}")
        
        # Preprocess text
        logger.info("Preprocessing text...")
        df['comment_text_clean'] = df['comment_text'].apply(self.preprocess_text)
        
        # Remove empty comments after preprocessing
        df = df[df['comment_text_clean'].str.len() > 0]
        logger.info(f"Data shape after removing empty comments: {df.shape}")
        
        # Prepare features and targets
        X = df['comment_text_clean'].values
        y = df[self.target_columns].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0]
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_vectorizer(self):
        """
        Create TF-IDF vectorizer
        
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        logger.info("Creating TF-IDF vectorizer...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            min_df=2,
            max_df=0.95
        )
        
        logger.info(f"Vectorizer configured with {self.max_features} max features")
        return self.vectorizer
    
    def build_model(self, model_type='xgboost'):
        """
        Build the machine learning model
        
        Args:
            model_type (str): Type of model ('xgboost', 'random_forest', 'logistic')
            
        Returns:
            Model: Configured model
        """
        logger.info(f"Building {model_type} model...")
        
        if model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic':
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Wrap in MultiOutputClassifier for multi-label classification
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        logger.info(f"{model_type} model built successfully")
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            
        Returns:
            Model: Trained model
        """
        logger.info("Starting model training...")
        
        # Fit vectorizer and transform training data
        logger.info("Vectorizing training data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info(f"Training data shape after vectorization: {X_train_tfidf.shape}")
        
        # Train model
        logger.info("Training the model...")
        self.model.fit(X_train_tfidf, y_train)
        
        logger.info("Training completed!")
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            logger.info("Evaluating on validation data...")
            X_val_tfidf = self.vectorizer.transform(X_val)
            val_score = self.model.score(X_val_tfidf, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics for each category
        results = {}
        
        for i, category in enumerate(self.target_columns):
            # Get probabilities for positive class
            if hasattr(y_pred_proba[i], 'shape') and y_pred_proba[i].shape[1] == 2:
                y_proba = y_pred_proba[i][:, 1]  # Probability of positive class
            else:
                y_proba = y_pred[:, i]  # Fallback to predictions
            
            # ROC AUC
            try:
                auc = roc_auc_score(y_test[:, i], y_proba)
            except:
                auc = 0.5  # Default for edge cases
            
            # Classification report
            report = classification_report(
                y_test[:, i], 
                y_pred[:, i], 
                output_dict=True,
                zero_division=0
            )
            
            results[category] = {
                'auc': auc,
                'precision': report['1']['precision'] if '1' in report else 0.0,
                'recall': report['1']['recall'] if '1' in report else 0.0,
                'f1': report['1']['f1-score'] if '1' in report else 0.0,
                'accuracy': report['accuracy']
            }
            
            logger.info(f"{category} - AUC: {auc:.4f}, F1: {results[category]['f1']:.4f}")
        
        # Overall metrics
        overall_auc = np.mean([results[cat]['auc'] for cat in self.target_columns])
        overall_f1 = np.mean([results[cat]['f1'] for cat in self.target_columns])
        
        logger.info(f"Overall AUC: {overall_auc:.4f}")
        logger.info(f"Overall F1: {overall_f1:.4f}")
        
        results['overall'] = {
            'auc': overall_auc,
            'f1': overall_f1
        }
        
        return results
    
    def plot_results(self, results):
        """
        Plot evaluation results
        
        Args:
            results: Results from evaluate_model
        """
        logger.info("Creating evaluation plots...")
        
        # Prepare data for plotting
        categories = self.target_columns
        auc_scores = [results[cat]['auc'] for cat in categories]
        f1_scores = [results[cat]['f1'] for cat in categories]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC scores
        axes[0].bar(categories, auc_scores, color='skyblue', alpha=0.7)
        axes[0].set_title('ROC AUC Scores by Category')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(auc_scores):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 scores
        axes[1].bar(categories, f1_scores, color='lightcoral', alpha=0.7)
        axes[1].set_title('F1 Scores by Category')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Evaluation plots saved as 'model_evaluation.png'")
    
    def save_model(self, model_path='toxicity_sklearn.pkl', vectorizer_path='vectorizer_sklearn.pkl'):
        """
        Save the trained model and vectorizer
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        logger.info("Saving model and vectorizer...")
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        # Save model configuration
        config = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'target_columns': self.target_columns,
            'vocab_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            'training_date': datetime.now().isoformat(),
            'model_type': 'scikit-learn',
            'vectorizer_type': 'TfidfVectorizer'
        }
        
        import json
        with open('model_config_sklearn.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Model configuration saved to model_config_sklearn.json")

def main():
    """Main training function"""
    logger.info("Starting toxicity detection model training (Scikit-Learn)...")
    logger.info("=" * 70)
    
    try:
        # Initialize trainer
        trainer = ToxicityModelTrainer(
            max_features=50000,
            ngram_range=(1, 2)
        )
        
        # Load data
        df = trainer.load_data('train.csv')
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # Create vectorizer
        trainer.create_vectorizer()
        
        # Build model (try XGBoost first, fallback to Random Forest)
        try:
            trainer.build_model('xgboost')
            model_type = 'XGBoost'
        except ImportError:
            logger.warning("XGBoost not available, using Random Forest")
            trainer.build_model('random_forest')
            model_type = 'Random Forest'
        
        # Split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        logger.info(f"Final training set: {len(X_train_final)}")
        logger.info(f"Validation set: {len(X_val)}")
        logger.info(f"Test set: {len(X_test)}")
        
        # Train model
        trainer.train_model(X_train_final, y_train_final, X_val, y_val)
        
        # Evaluate model
        results = trainer.evaluate_model(X_test, y_test)
        
        # Plot results
        trainer.plot_results(results)
        
        # Save model
        trainer.save_model()
        
        logger.info("=" * 70)
        logger.info("🎉 Training completed successfully!")
        logger.info("=" * 70)
        
        logger.info("Files created:")
        logger.info("- toxicity_sklearn.pkl (trained model)")
        logger.info("- vectorizer_sklearn.pkl (TF-IDF vectorizer)")
        logger.info("- model_config_sklearn.json (configuration)")
        logger.info("- model_evaluation.png (evaluation plots)")
        logger.info("- training_sklearn.log (training log)")
        
        logger.info("\nNext steps:")
        logger.info("1. Run 'python app_sklearn.py' to start the web application")
        logger.info("2. Access the web interface at http://localhost:5000")
        
        # Print final performance summary
        logger.info(f"\n📊 Final {model_type} Model Performance:")
        logger.info("-" * 40)
        for category in trainer.target_columns:
            auc = results[category]['auc']
            f1 = results[category]['f1']
            logger.info(f"{category:15}: AUC={auc:.3f}, F1={f1:.3f}")
        
        logger.info("-" * 40)
        logger.info(f"{'Overall':15}: AUC={results['overall']['auc']:.3f}, F1={results['overall']['f1']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Please check the error details above.")
        sys.exit(1)

if __name__ == "__main__":
    main()