"""
Auto-Flexible AI System: ML + DL + LLM with Hugging Face Integration
Automatically detects target column and adapts to any dataset
Output: Clean JSON format only - ENHANCED VERSION WITH TEXT SUPPORT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import requests
import warnings
import os
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

class AutoFlexibleAISystem:
    def __init__(self, hf_api_key: str = None):
        """
        Initialize the auto-flexible AI system with Hugging Face API
        
        Args:
            hf_api_key (str): Hugging Face API key (optional if set in .env)
        """
        # Try to get API key from .env if not provided
        self.hf_api_key = hf_api_key or os.getenv('HF_API_KEY') or os.getenv('HUGGINGFACE_API_KEY')
        
        if not self.hf_api_key:
            print("⚠️ Warning: No Hugging Face API key found. LLM explanations will use fallback.")
        
        self.ml_model = None
        self.dl_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.original_feature_names = None
        self.target_names = None
        self.dataset_info = {}
        self.has_text_features = False
        self.text_feature_columns = []
        
        # Hugging Face API setup
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
        
    def auto_detect_target(self, data: pd.DataFrame) -> str:
        """
        Automatically detect target column (assumes last column is target)
        """
        target_column = data.columns[-1]
        print(f"🎯 Auto-detected target column: '{target_column}'")
        return target_column
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data before processing
        """
        print("🔍 Validating dataset...")
        
        # Remove empty rows
        initial_rows = len(data)
        data = data.dropna(how='all')
        if len(data) < initial_rows:
            print(f"  ✓ Removed {initial_rows - len(data)} completely empty rows")
        
        # Remove columns that are entirely empty
        initial_cols = len(data.columns)
        data = data.dropna(axis=1, how='all')
        if len(data.columns) < initial_cols:
            print(f"  ✓ Removed {initial_cols - len(data.columns)} empty columns")
        
        # Check for sufficient data
        if len(data) < 10:
            raise ValueError("Dataset too small (less than 10 rows)")
        
        if len(data.columns) < 2:
            raise ValueError("Dataset needs at least 2 columns (features + target)")
        
        print(f"✅ Data validation passed: {len(data)} rows, {len(data.columns)} columns")
        return data
    
    def detect_text_columns(self, data: pd.DataFrame, target_column: str) -> List[str]:
        """
        Detect columns that contain text data (long strings)
        """
        text_columns = []
        for col in data.columns:
            if col == target_column:
                continue
            if data[col].dtype == 'object':
                # Check average string length
                avg_length = data[col].astype(str).str.len().mean()
                unique_ratio = len(data[col].unique()) / len(data)
                
                # If high unique ratio and long strings, it's likely text
                if avg_length > 20 or unique_ratio > 0.5:
                    text_columns.append(col)
                    
        return text_columns
    
    def analyze_dataset(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Enhanced dataset analysis
        """
        # Detect text columns first
        self.text_feature_columns = self.detect_text_columns(data, target_column)
        self.has_text_features = len(self.text_feature_columns) > 0
        
        analysis = {
            'name': 'Auto-Dataset',
            'samples': len(data),
            'features': len(data.columns) - 1,
            'target_column': target_column,
            'feature_names': [col for col in data.columns if col != target_column],
            'target_classes': sorted(data[target_column].unique()),
            'numeric_features': [],
            'categorical_features': [],
            'text_features': self.text_feature_columns,
            'missing_values': data.isnull().sum().to_dict(),
            'feature_types': {},
            'data_types': {}
        }
        
        # Analyze feature types and data types
        for col in data.columns:
            dtype_str = str(data[col].dtype)
            analysis['data_types'][col] = dtype_str
            
            if col != target_column:
                if col in self.text_feature_columns:
                    analysis['feature_types'][col] = 'text'
                elif data[col].dtype in ['int64', 'float64']:
                    analysis['numeric_features'].append(col)
                    analysis['feature_types'][col] = 'numeric'
                else:
                    analysis['categorical_features'].append(col)
                    analysis['feature_types'][col] = 'categorical'
        
        print(f"📊 Dataset analysis:")
        print(f"   Numeric features: {len(analysis['numeric_features'])}")
        print(f"   Categorical features: {len(analysis['categorical_features'])}")
        print(f"   Text features: {len(analysis['text_features'])}")
        if self.has_text_features:
            print(f"   ⚠️  Text columns detected: {self.text_feature_columns}")
        print(f"   Missing values: {sum(analysis['missing_values'].values())} total")
        
        self.dataset_info = analysis
        return analysis
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically preprocess any dataset (auto-detects target) - ENHANCED
        """
        print("\n📊 Auto-preprocessing dataset...")
        processed_data = data.copy()
        
        # Auto-detect target column
        target_column = self.auto_detect_target(processed_data)
        
        # Store original feature names
        self.original_feature_names = [col for col in processed_data.columns if col != target_column]
        
        # Analyze dataset
        self.analyze_dataset(processed_data, target_column)
        
        # Handle missing values FIRST
        print("🔧 Handling missing values...")
        for col in processed_data.columns:
            if processed_data[col].isnull().sum() > 0:
                if processed_data[col].dtype in ['int64', 'float64']:
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                else:
                    mode_value = processed_data[col].mode()
                    if len(mode_value) > 0:
                        processed_data[col].fillna(mode_value[0], inplace=True)
                    else:
                        processed_data[col].fillna('unknown', inplace=True)
                print(f"  ✓ Fixed missing values in {col}")
        
        # Convert ALL non-numeric columns to numeric
        print("🎯 Converting all columns to numeric...")
        
        for col in processed_data.columns:
            # Skip if already numeric
            if processed_data[col].dtype in ['int64', 'float64']:
                print(f"  ✓ {col} already numeric")
                continue
            
            # Try to convert to numeric first
            try:
                temp_col = pd.to_numeric(processed_data[col], errors='coerce')
                if temp_col.isnull().sum() == 0:
                    processed_data[col] = temp_col
                    print(f"  ✓ Converted {col} to numeric")
                    continue
            except:
                pass
            
            # Use label encoding for categorical/text columns
            print(f"  🔄 Label encoding {col}...")
            le = LabelEncoder()
            
            # Convert to string first to handle mixed types
            processed_data[col] = processed_data[col].astype(str)
            
            # Replace any remaining NaN strings
            processed_data[col] = processed_data[col].replace(['nan', 'NaN', 'None', ''], 'unknown')
            
            # Apply label encoding
            processed_data[col] = le.fit_transform(processed_data[col])
            self.label_encoders[col] = le
            
            print(f"    ✓ Encoded {col} ({len(le.classes_)} unique values)")
        
        # Ensure target column is properly encoded
        target_unique = len(processed_data[target_column].unique())
        print(f"🎯 Target column '{target_column}' has {target_unique} unique values")
        
        # Update target names based on encoding
        if target_column in self.label_encoders:
            self.dataset_info['target_names'] = self.label_encoders[target_column].classes_.tolist()
        else:
            unique_targets = sorted(processed_data[target_column].unique())
            if len(unique_targets) == 2:
                self.dataset_info['target_names'] = ['Class_0', 'Class_1']
            else:
                self.dataset_info['target_names'] = [f'Class_{i}' for i in unique_targets]
        
        # Set class attributes
        self.feature_names = [col for col in processed_data.columns if col != target_column]
        self.target_names = self.dataset_info['target_names']
        
        # Final validation
        print("🔍 Final validation...")
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                print(f"  ⚠️  Warning: {col} still contains non-numeric data, forcing conversion...")
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                processed_data[col].fillna(0, inplace=True)
        
        # Check for any remaining issues
        non_numeric_cols = processed_data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Failed to convert columns to numeric: {list(non_numeric_cols)}")
        
        print(f"✅ Enhanced preprocessing complete!")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Classes: {len(self.target_names)} {self.target_names}")
        print(f"   Samples: {len(processed_data)}")
        print(f"   All columns numeric: ✓")
        
        return processed_data
    
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and prepare any CSV dataset (auto-detects everything) - ENHANCED
        """
        print(f"\n📁 Loading dataset: {csv_path}")
        
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            data = None
            
            for encoding in encodings_to_try:
                try:
                    data = pd.read_csv(csv_path, encoding=encoding)
                    print(f"✓ Loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                raise ValueError("Could not read CSV file with any encoding")
                
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {str(e)}")
        
        print(f"✓ Raw data: {data.shape[0]} samples, {data.shape[1]} columns")
        
        # Validate data
        data = self.validate_data(data)
        
        # Auto-preprocess the data
        processed_data = self.preprocess_data(data)
        
        return processed_data
    
    def prepare_for_training(self, data: pd.DataFrame):
        """
        Prepare data for training
        """
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_ml_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train Random Forest model (automatically adapts to data)
        """
        print("\n🌳 Training ML Model (Random Forest)...")
        
        n_estimators = min(200, max(50, len(X_train) // 10))
        max_depth = min(20, max(3, int(np.log2(len(self.feature_names)) + 2)))
        
        self.ml_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.ml_model.fit(X_train, y_train)
        print(f"✅ ML Model trained! (n_estimators={n_estimators}, max_depth={max_depth})")
    
    def train_dl_model(self, X_train: np.ndarray, y_train: np.ndarray, num_classes: int):
        """
        Train Neural Network (automatically adapts architecture)
        """
        print("\n🧠 Training DL Model (Neural Network)...")
        
        input_dim = X_train.shape[1]
        hidden_units = [
            min(128, max(16, input_dim * 2)),
            min(64, max(8, input_dim)),
            min(32, max(4, input_dim // 2))
        ]
        
        if num_classes == 2:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_units = num_classes
            activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'
        
        self.dl_model = keras.Sequential([
            layers.Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(hidden_units[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_units[2], activation='relu'),
            layers.Dense(output_units, activation=activation)
        ])
        
        self.dl_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        epochs = min(100, max(20, 1000 // (len(X_train) // 100 + 1)))
        batch_size = min(64, max(8, len(X_train) // 100))
        
        self.dl_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2
        )
        
        print(f"✅ DL Model trained! (epochs={epochs}, batch_size={batch_size})")
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate both models
        """
        print("\n📈 EVALUATING MODELS")
        print("=" * 60)
        
        ml_pred = self.ml_model.predict(X_test)
        
        dl_pred_probs = self.dl_model.predict(X_test, verbose=0)
        
        if len(self.target_names) == 2:
            if dl_pred_probs.shape[1] == 1:
                dl_pred = (dl_pred_probs > 0.5).astype(int).flatten()
            else:
                dl_pred = np.argmax(dl_pred_probs, axis=1)
        else:
            dl_pred = np.argmax(dl_pred_probs, axis=1)
        
        average_method = 'binary' if len(self.target_names) == 2 else 'weighted'
        
        ml_metrics = {
            'accuracy': accuracy_score(y_test, ml_pred),
            'precision': precision_score(y_test, ml_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_test, ml_pred, average=average_method, zero_division=0),
            'f1_score': f1_score(y_test, ml_pred, average=average_method, zero_division=0)
        }
        
        dl_metrics = {
            'accuracy': accuracy_score(y_test, dl_pred),
            'precision': precision_score(y_test, dl_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_test, dl_pred, average=average_method, zero_division=0),
            'f1_score': f1_score(y_test, dl_pred, average=average_method, zero_division=0)
        }
        
        print(f"\n🌳 ML Model (Random Forest):")
        for metric, value in ml_metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        print(f"\n🧠 DL Model (Neural Network):")
        for metric, value in dl_metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        print("=" * 60)
        
        return ml_metrics, dl_metrics
    
    def encode_input(self, raw_input: List[str]) -> List[float]:
        """
        Encode raw input values using the same encoders used during training
        """
        encoded_values = []
        
        for i, (col_name, value) in enumerate(zip(self.feature_names, raw_input)):
            value = str(value).strip()
            
            if col_name in self.label_encoders:
                le = self.label_encoders[col_name]
                # Check if value exists in encoder
                if value in le.classes_:
                    encoded_value = le.transform([value])[0]
                else:
                    # Unknown value - use most common class index or 0
                    print(f"  ⚠️ Unknown value for {col_name}: '{value[:30]}...', using default")
                    encoded_value = 0
                encoded_values.append(float(encoded_value))
            else:
                # Try to convert to numeric
                try:
                    encoded_values.append(float(value))
                except ValueError:
                    print(f"  ⚠️ Cannot convert '{value}' to numeric, using 0")
                    encoded_values.append(0.0)
        
        return encoded_values
    
    def predict_single(self, input_data: List[float]):
        """
        Make prediction for single input
        """
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        
        ml_prediction = self.ml_model.predict(input_scaled)[0]
        ml_proba = self.ml_model.predict_proba(input_scaled)[0]
        ml_confidence = float(np.max(ml_proba))
        
        dl_proba = self.dl_model.predict(input_scaled, verbose=0)[0]
        
        if len(self.target_names) == 2:
            if len(dl_proba) == 1:
                dl_prediction = int(dl_proba[0] > 0.5)
                dl_confidence = float(max(dl_proba[0], 1 - dl_proba[0]))
            else:
                dl_prediction = int(np.argmax(dl_proba))
                dl_confidence = float(np.max(dl_proba))
        else:
            dl_prediction = int(np.argmax(dl_proba))
            dl_confidence = float(np.max(dl_proba))
        
        if ml_confidence >= dl_confidence:
            final_prediction = int(ml_prediction)
            final_confidence = ml_confidence
            chosen_model = "ML (Random Forest)"
        else:
            final_prediction = dl_prediction
            final_confidence = dl_confidence
            chosen_model = "DL (Neural Network)"
        
        return {
            'ml_prediction': int(ml_prediction),
            'ml_confidence': ml_confidence,
            'dl_prediction': dl_prediction,
            'dl_confidence': dl_confidence,
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'chosen_model': chosen_model
        }
    
    def generate_explanation(self, input_data: List[float], prediction_result: Dict, 
                           ml_metrics: Dict, dl_metrics: Dict, raw_input: List[str] = None) -> str:
        """
        Generate dynamic explanation based on actual prediction results
        """
        predicted_class = self.target_names[prediction_result['final_prediction']]
        chosen_model = prediction_result['chosen_model']
        final_confidence = prediction_result['final_confidence']
        ml_conf = prediction_result['ml_confidence']
        dl_conf = prediction_result['dl_confidence']
        ml_pred = prediction_result['ml_prediction']
        dl_pred = prediction_result['dl_prediction']
        
        # Determine why this model was selected
        if chosen_model == "ML (Random Forest)":
            other_model = "DL (Neural Network)"
            confidence_diff = ml_conf - dl_conf
        else:
            other_model = "ML (Random Forest)"
            confidence_diff = dl_conf - ml_conf
        
        # Create dynamic explanation parts
        explanation_parts = []
        
        # Part 1: What was predicted
        explanation_parts.append(
            f"The {chosen_model} predicted '{predicted_class}' with {final_confidence:.1%} confidence."
        )
        
        # Part 2: Why this model was selected
        if confidence_diff > 0.2:
            explanation_parts.append(
                f"This model was selected because it showed significantly higher confidence "
                f"({confidence_diff:.1%} more) compared to the {other_model}."
            )
        elif confidence_diff > 0.05:
            explanation_parts.append(
                f"This model was chosen as it demonstrated {confidence_diff:.1%} higher confidence than the {other_model}."
            )
        else:
            explanation_parts.append(
                f"Both models showed similar confidence levels, but {chosen_model} had a slight edge."
            )
        
        # Part 3: Model agreement/disagreement
        if ml_pred == dl_pred:
            explanation_parts.append(
                f"Both ML and DL models agreed on this prediction, increasing reliability."
            )
        else:
            ml_class = self.target_names[ml_pred]
            dl_class = self.target_names[dl_pred]
            explanation_parts.append(
                f"Note: Models disagreed - ML predicted '{ml_class}' ({ml_conf:.1%}) while DL predicted '{dl_class}' ({dl_conf:.1%}). "
                f"The higher confidence prediction was selected."
            )
        
        # Part 4: Confidence level assessment
        if final_confidence >= 0.9:
            explanation_parts.append("High confidence prediction - very reliable result.")
        elif final_confidence >= 0.7:
            explanation_parts.append("Good confidence level - reasonably reliable prediction.")
        elif final_confidence >= 0.5:
            explanation_parts.append("Moderate confidence - prediction should be verified if critical.")
        else:
            explanation_parts.append("Low confidence - consider this prediction with caution.")
        
        # Part 5: Model performance context
        if chosen_model == "ML (Random Forest)":
            explanation_parts.append(
                f"The Random Forest model has {ml_metrics['accuracy']:.1%} overall accuracy "
                f"and {ml_metrics['f1_score']:.2f} F1-score on test data."
            )
        else:
            explanation_parts.append(
                f"The Neural Network model has {dl_metrics['accuracy']:.1%} overall accuracy "
                f"and {dl_metrics['f1_score']:.2f} F1-score on test data."
            )
        
        # Combine all parts
        full_explanation = " ".join(explanation_parts)
        
        return full_explanation
    
    def get_final_output(self, input_data: List[float], ml_metrics: Dict, dl_metrics: Dict, 
                        raw_input: List[str] = None) -> Dict:
        """
        Generate final JSON output in the exact expected format
        """
        prediction_result = self.predict_single(input_data)
        
        # Generate dynamic explanation with all context
        explanation = self.generate_explanation(
            input_data, prediction_result, ml_metrics, dl_metrics, raw_input
        )
        
        predicted_class = self.target_names[prediction_result['final_prediction']]
        
        ml_vs_dl = (
            f"ML Model: {prediction_result['ml_confidence']:.2%} confidence | "
            f"DL Model: {prediction_result['dl_confidence']:.2%} confidence | "
            f"Selected: {prediction_result['chosen_model']}"
        )
        
        return {
            "prediction": predicted_class,
            "confidence": f"{prediction_result['final_confidence']:.2%}",
            "ml_vs_dl_comparison": ml_vs_dl,
            "llm_explanation": explanation
        }
    
    def parse_input_with_text(self, raw_input: str) -> List[str]:
        """
        Parse input that may contain text with commas
        Format: value1 | value2 | text with, commas | value3
        """
        # Check if using pipe separator (easier for text)
        if '|' in raw_input:
            parts = [p.strip() for p in raw_input.split('|')]
        else:
            # Try to handle quoted strings with commas
            parts = []
            current = ""
            in_quotes = False
            
            for char in raw_input:
                if char == '"' or char == "'":
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current.strip())
                    current = ""
                else:
                    current += char
            
            if current:
                parts.append(current.strip())
        
        return parts


def main():
    """
    Main function - super simple, just CSV path needed (API key from .env)
    """
    print("\n" + "🚀" + "="*58)
    print("   AUTO-FLEXIBLE AI SYSTEM - ENV CONFIG ENABLED")
    print("="*60 + "\n")
    
    # Check for API key in environment
    hf_api_key = os.getenv('HF_API_KEY') or os.getenv('HUGGINGFACE_API_KEY')
    
    if hf_api_key:
        print("✅ Hugging Face API key loaded from .env file")
    else:
        print("⚠️ No API key found in .env file")
        hf_api_key = input("🤗 Enter Hugging Face API key (or press Enter to skip): ").strip()
    
    csv_path = input("📁 Enter CSV file path: ").strip()
    
    if not csv_path:
        print("❌ Error: CSV path is required!")
        return
    
    try:
        print(f"\n🔧 Initializing enhanced auto-system...")
        ai_system = AutoFlexibleAISystem(hf_api_key=hf_api_key)
        
        data = ai_system.load_and_prepare_data(csv_path)
        
        print(f"\n📊 Auto-Analysis Complete:")
        print(f"   📈 Samples: {ai_system.dataset_info['samples']}")
        print(f"   📋 Features: {ai_system.dataset_info['features']}")
        print(f"   🏷️  Target: {ai_system.dataset_info['target_column']}")
        print(f"   📝 Classes: {', '.join(map(str, ai_system.target_names))}")
        
        if ai_system.has_text_features:
            print(f"   📄 Text columns: {', '.join(ai_system.text_feature_columns)}")
        
        X_train, X_test, y_train, y_test = ai_system.prepare_for_training(data)
        ai_system.train_ml_model(X_train, y_train)
        ai_system.train_dl_model(X_train, y_train, len(ai_system.target_names))
        ml_metrics, dl_metrics = ai_system.evaluate_models(X_test, y_test)
        
        print(f"\n🎮 INTERACTIVE PREDICTION - JSON OUTPUT")
        print("="*60)
        print(f"Features: {', '.join(ai_system.original_feature_names)}")
        
        if ai_system.has_text_features:
            print("\n💡 TIP: Use | (pipe) to separate values when text contains commas")
            print("   Example: 123 | Some text here | more text | 456")
        
        print("\nType 'quit' to exit\n")
        
        while True:
            user_input = input("🔮 Enter feature values: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Session ended!")
                break
            
            try:
                # Parse input (handles text with commas)
                raw_values = ai_system.parse_input_with_text(user_input)
                
                if len(raw_values) != len(ai_system.feature_names):
                    print(f"❌ Error: Need exactly {len(ai_system.feature_names)} values, got {len(raw_values)}")
                    continue
                
                print("\n🔄 Processing...")
                
                # Encode the raw input using label encoders
                encoded_input = ai_system.encode_input(raw_values)
                
                # Pass raw_values for better explanation context
                output = ai_system.get_final_output(encoded_input, ml_metrics, dl_metrics, raw_values)
                
                print(json.dumps(output, indent=2))
                print()
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ System Error: {str(e)}")


if __name__ == "__main__":
    main()