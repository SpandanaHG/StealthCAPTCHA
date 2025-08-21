import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
from app import db
from models import ModelMetrics

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "1.0"
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, behavioral_data):
        """Extract features from behavioral data for ML prediction"""
        features = []
        
        # Mouse movement features
        mouse_movements = behavioral_data.get('mouse_movements', [])
        if mouse_movements:
            # Calculate velocities
            velocities = []
            for i in range(1, len(mouse_movements)):
                prev = mouse_movements[i-1]
                curr = mouse_movements[i]
                time_diff = (curr['timestamp'] - prev['timestamp']) / 1000.0  # Convert to seconds
                if time_diff > 0:
                    distance = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
                    velocity = distance / time_diff
                    velocities.append(velocity)
            
            if velocities:
                features.extend([
                    np.mean(velocities),  # Average velocity
                    np.std(velocities),   # Velocity standard deviation
                    len(velocities),      # Number of movements
                    np.max(velocities) if velocities else 0,  # Max velocity
                    np.min(velocities) if velocities else 0   # Min velocity
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Click pattern features
        click_patterns = behavioral_data.get('click_patterns', [])
        if click_patterns:
            # Click intervals
            intervals = []
            for i in range(1, len(click_patterns)):
                interval = (click_patterns[i]['timestamp'] - click_patterns[i-1]['timestamp']) / 1000.0
                intervals.append(interval)
            
            features.extend([
                len(click_patterns),  # Number of clicks
                np.mean(intervals) if intervals else 0,  # Average click interval
                np.std(intervals) if intervals else 0,   # Click interval variance
            ])
        else:
            features.extend([0, 0, 0])
        
        # Keystroke pattern features
        keystroke_patterns = behavioral_data.get('keystroke_patterns', [])
        if keystroke_patterns:
            # Dwell times (key press duration)
            dwell_times = [k.get('duration', 0) for k in keystroke_patterns]
            
            # Flight times (time between key presses)
            flight_times = []
            for i in range(1, len(keystroke_patterns)):
                flight_time = keystroke_patterns[i]['timestamp'] - keystroke_patterns[i-1]['timestamp']
                flight_times.append(flight_time)
            
            features.extend([
                len(keystroke_patterns),  # Number of keystrokes
                np.mean(dwell_times) if dwell_times else 0,  # Average dwell time
                np.std(dwell_times) if dwell_times else 0,   # Dwell time variance
                np.mean(flight_times) if flight_times else 0,  # Average flight time
                np.std(flight_times) if flight_times else 0,   # Flight time variance
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Scroll pattern features
        scroll_patterns = behavioral_data.get('scroll_patterns', [])
        if scroll_patterns:
            scroll_speeds = []
            for scroll in scroll_patterns:
                speed = abs(scroll.get('deltaY', 0))
                scroll_speeds.append(speed)
            
            features.extend([
                len(scroll_patterns),  # Number of scroll events
                np.mean(scroll_speeds) if scroll_speeds else 0,  # Average scroll speed
                np.std(scroll_speeds) if scroll_speeds else 0,   # Scroll speed variance
            ])
        else:
            features.extend([0, 0, 0])
        
        # Device/Browser features (encoded as numerical values)
        user_agent = behavioral_data.get('user_agent', '')
        features.extend([
            1 if 'Chrome' in user_agent else 0,
            1 if 'Firefox' in user_agent else 0,
            1 if 'Safari' in user_agent else 0,
            1 if 'Mobile' in user_agent else 0,
            len(user_agent),  # User agent length
        ])
        
        # Screen resolution features
        screen_res = behavioral_data.get('screen_resolution', '0x0')
        try:
            width, height = map(int, screen_res.split('x'))
            features.extend([width, height, width * height])
        except:
            features.extend([0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data for initial model training"""
        logging.info("Generating synthetic training data...")
        
        X = []
        y = []
        
        # Generate human-like patterns
        for _ in range(num_samples // 2):
            # Human mouse movements: more natural, varied velocities
            mouse_velocity_avg = np.random.normal(150, 50)
            mouse_velocity_std = np.random.normal(75, 25)
            num_movements = np.random.randint(10, 100)
            max_velocity = mouse_velocity_avg + np.random.normal(200, 50)
            min_velocity = max(0, mouse_velocity_avg - np.random.normal(100, 30))
            
            # Human click patterns: more irregular intervals
            num_clicks = np.random.randint(1, 20)
            click_interval_avg = np.random.normal(2.0, 1.0)
            click_interval_std = np.random.normal(1.5, 0.5)
            
            # Human keystroke patterns: natural typing rhythm
            num_keystrokes = np.random.randint(0, 50)
            dwell_time_avg = np.random.normal(100, 30)
            dwell_time_std = np.random.normal(50, 15)
            flight_time_avg = np.random.normal(150, 50)
            flight_time_std = np.random.normal(100, 30)
            
            # Human scroll patterns: varied speeds
            num_scrolls = np.random.randint(0, 30)
            scroll_speed_avg = np.random.normal(50, 20)
            scroll_speed_std = np.random.normal(30, 10)
            
            # Browser features (realistic distribution)
            chrome = np.random.choice([0, 1], p=[0.3, 0.7])
            firefox = np.random.choice([0, 1], p=[0.8, 0.2]) if not chrome else 0
            safari = np.random.choice([0, 1], p=[0.9, 0.1]) if not chrome and not firefox else 0
            mobile = np.random.choice([0, 1], p=[0.6, 0.4])
            ua_length = np.random.randint(80, 200)
            
            # Screen resolution
            common_widths = [1920, 1366, 1440, 1280, 1024]
            common_heights = [1080, 768, 900, 720, 768]
            width = np.random.choice(common_widths)
            height = np.random.choice(common_heights)
            
            features = [
                mouse_velocity_avg, mouse_velocity_std, num_movements, max_velocity, min_velocity,
                num_clicks, click_interval_avg, click_interval_std,
                num_keystrokes, dwell_time_avg, dwell_time_std, flight_time_avg, flight_time_std,
                num_scrolls, scroll_speed_avg, scroll_speed_std,
                chrome, firefox, safari, mobile, ua_length,
                width, height, width * height
            ]
            
            X.append(features)
            y.append(1)  # Human
        
        # Generate bot-like patterns
        for _ in range(num_samples // 2):
            # Bot mouse movements: more uniform, mechanical
            mouse_velocity_avg = np.random.normal(300, 20)  # Higher, more consistent
            mouse_velocity_std = np.random.normal(10, 5)    # Lower variance
            num_movements = np.random.randint(50, 200)      # More movements
            max_velocity = mouse_velocity_avg + np.random.normal(50, 10)
            min_velocity = mouse_velocity_avg - np.random.normal(50, 10)
            
            # Bot click patterns: very regular intervals
            num_clicks = np.random.randint(5, 50)
            click_interval_avg = np.random.normal(0.5, 0.1)  # Very fast, regular
            click_interval_std = np.random.normal(0.05, 0.02)  # Very low variance
            
            # Bot keystroke patterns: mechanical typing
            num_keystrokes = np.random.randint(10, 100)
            dwell_time_avg = np.random.normal(50, 5)        # Consistent, short
            dwell_time_std = np.random.normal(5, 2)         # Very low variance
            flight_time_avg = np.random.normal(50, 5)       # Very consistent
            flight_time_std = np.random.normal(5, 2)        # Very low variance
            
            # Bot scroll patterns: mechanical scrolling
            num_scrolls = np.random.randint(0, 10)
            scroll_speed_avg = np.random.normal(100, 5)     # Very consistent
            scroll_speed_std = np.random.normal(5, 2)       # Low variance
            
            # Browser features (bot-like patterns)
            chrome = np.random.choice([0, 1], p=[0.1, 0.9])  # Bots often use Chrome
            firefox = 0
            safari = 0
            mobile = np.random.choice([0, 1], p=[0.9, 0.1])  # Usually desktop
            ua_length = np.random.randint(50, 120)           # Shorter UA strings
            
            # Screen resolution (common bot resolutions)
            width = np.random.choice([1920, 1366, 1024])
            height = np.random.choice([1080, 768, 768])
            
            features = [
                mouse_velocity_avg, mouse_velocity_std, num_movements, max_velocity, min_velocity,
                num_clicks, click_interval_avg, click_interval_std,
                num_keystrokes, dwell_time_avg, dwell_time_std, flight_time_avg, flight_time_std,
                num_scrolls, scroll_speed_avg, scroll_speed_std,
                chrome, firefox, safari, mobile, ua_length,
                width, height, width * height
            ]
            
            X.append(features)
            y.append(0)  # Bot
        
        return np.array(X), np.array(y)
    
    def train_initial_model(self):
        """Train the initial model with synthetic data"""
        if self.is_trained:
            return
        
        try:
            # Generate training data
            X, y = self.generate_training_data(2000)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logging.info(f"Model training completed:")
            logging.info(f"Accuracy: {accuracy:.3f}")
            logging.info(f"Precision: {precision:.3f}")
            logging.info(f"Recall: {recall:.3f}")
            logging.info(f"F1-score: {f1:.3f}")
            
            # Save model metrics to database
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_samples=len(X),
                human_samples=len(y[y == 1]),
                bot_samples=len(y[y == 0]),
                model_version=self.model_version
            )
            
            db.session.add(metrics)
            db.session.commit()
            
            self.is_trained = True
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
    
    def predict(self, features):
        """Make a prediction on behavioral features"""
        if not self.is_trained:
            self.train_initial_model()
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            # Get confidence (probability of predicted class)
            confidence = prediction_proba[prediction]
            
            # Convert to human-readable format
            prediction_label = 'human' if prediction == 1 else 'bot'
            
            return prediction_label, float(confidence)
        
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return 'unknown', 0.5
    
    def save_model(self):
        """Save the trained model and scaler to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/stealth_captcha_model.pkl')
            joblib.dump(self.scaler, 'models/stealth_captcha_scaler.pkl')
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the trained model and scaler from disk"""
        try:
            if os.path.exists('models/stealth_captcha_model.pkl') and os.path.exists('models/stealth_captcha_scaler.pkl'):
                self.model = joblib.load('models/stealth_captcha_model.pkl')
                self.scaler = joblib.load('models/stealth_captcha_scaler.pkl')
                self.is_trained = True
                logging.info("Model loaded successfully")
            else:
                logging.info("No saved model found, will train new model")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
