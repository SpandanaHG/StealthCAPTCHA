from app import db
from datetime import datetime
from sqlalchemy import func

class BehavioralData(db.Model):
    __tablename__ = 'behavioral_data'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Mouse movement data
    mouse_movements = db.Column(db.JSON)  # Array of {x, y, timestamp}
    click_patterns = db.Column(db.JSON)   # Array of {x, y, timestamp, button}
    scroll_patterns = db.Column(db.JSON)  # Array of {deltaX, deltaY, timestamp}
    
    # Typing patterns
    keystroke_patterns = db.Column(db.JSON)  # Array of {key, timestamp, duration}
    
    # Browser/Device fingerprint
    user_agent = db.Column(db.Text)
    screen_resolution = db.Column(db.String(20))
    timezone = db.Column(db.String(50))
    language = db.Column(db.String(10))
    platform = db.Column(db.String(50))
    
    # Behavioral metrics (calculated)
    mouse_velocity_avg = db.Column(db.Float)
    mouse_velocity_std = db.Column(db.Float)
    click_frequency = db.Column(db.Float)
    typing_rhythm_consistency = db.Column(db.Float)
    
    # IP and network data
    ip_address = db.Column(db.String(45))
    
    # Classification result
    is_human = db.Column(db.Boolean)
    confidence_score = db.Column(db.Float)
    
    def __repr__(self):
        return f'<BehavioralData {self.session_id}>'

class DetectionLog(db.Model):
    __tablename__ = 'detection_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Detection results
    prediction = db.Column(db.String(10))  # 'human' or 'bot'
    confidence = db.Column(db.Float)
    
    # Context
    page_url = db.Column(db.String(500))
    action_type = db.Column(db.String(50))  # 'form_submit', 'login', 'payment', etc.
    
    # Additional metadata
    processing_time_ms = db.Column(db.Integer)
    
    def __repr__(self):
        return f'<DetectionLog {self.session_id}: {self.prediction}>'

class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Model performance metrics
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    
    # Training data stats
    training_samples = db.Column(db.Integer)
    human_samples = db.Column(db.Integer)
    bot_samples = db.Column(db.Integer)
    
    # Model version
    model_version = db.Column(db.String(20))
    
    def __repr__(self):
        return f'<ModelMetrics v{self.model_version}: {self.accuracy:.3f}>'
