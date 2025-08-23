
from flask import render_template, request, jsonify, session
from app import app, db
from models import BehavioralData, DetectionLog, ModelMetrics
from behavioral_analyzer import BehavioralAnalyzer
from ml_model import MLModel
import uuid
import time
from datetime import datetime, timedelta
import logging

# Initialize components
behavioral_analyzer = BehavioralAnalyzer()
ml_model = MLModel()

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/research')
def research():
    """Research interface for behavioral biometrics testing"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('research.html', session_id=session['session_id'])

@app.route('/methodology')
def methodology():
    """Research methodology and technical details"""
    return render_template('methodology.html')

@app.route('/analytics')
def analytics():
    """Research analytics dashboard for viewing detection results"""
    # Get recent detection statistics
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    
    # Recent detections
    recent_detections = DetectionLog.query.filter(
        DetectionLog.timestamp >= week_ago
    ).order_by(DetectionLog.timestamp.desc()).limit(100).all()
    
    # Statistics
    total_detections = DetectionLog.query.filter(
        DetectionLog.timestamp >= week_ago
    ).count()
    
    human_detections = DetectionLog.query.filter(
        DetectionLog.timestamp >= week_ago,
        DetectionLog.prediction == 'human'
    ).count()
    
    bot_detections = DetectionLog.query.filter(
        DetectionLog.timestamp >= week_ago,
        DetectionLog.prediction == 'bot'
    ).count()
    
    # Model metrics
    latest_metrics = ModelMetrics.query.order_by(
        ModelMetrics.timestamp.desc()
    ).first()
    
    stats = {
        'total_detections': total_detections,
        'human_detections': human_detections,
        'bot_detections': bot_detections,
        'human_percentage': (human_detections / total_detections * 100) if total_detections > 0 else 0,
        'bot_percentage': (bot_detections / total_detections * 100) if total_detections > 0 else 0
    }
    
    return render_template('analytics.html', 
                         recent_detections=recent_detections,
                         stats=stats,
                         model_metrics=latest_metrics)

@app.route('/api/behavioral_data', methods=['POST'])
def collect_behavioral_data():
    """Endpoint to collect behavioral data from frontend"""
    try:
        data = request.get_json()
        
        if not data or 'sessionId' not in data:
            return jsonify({'error': 'Invalid data'}), 400
        
        session_id = data['sessionId']
        
        # Create new behavioral data record
        behavioral_data = BehavioralData(
            session_id=session_id,
            mouse_movements=data.get('mouseMovements', []),
            click_patterns=data.get('clickPatterns', []),
            scroll_patterns=data.get('scrollPatterns', []),
            keystroke_patterns=data.get('keystrokePatterns', []),
            user_agent=request.headers.get('User-Agent'),
            screen_resolution=data.get('deviceFingerprint', {}).get('screenResolution'),
            timezone=data.get('deviceFingerprint', {}).get('timezone'),
            language=data.get('deviceFingerprint', {}).get('language'),
            platform=data.get('deviceFingerprint', {}).get('platform'),
            ip_address=request.remote_addr
        )
        
        # Analyze behavioral patterns
        metrics = behavioral_analyzer.analyze_patterns(data)
        behavioral_data.mouse_velocity_avg = metrics.get('mouse_velocity_avg')
        behavioral_data.mouse_velocity_std = metrics.get('mouse_velocity_std')
        behavioral_data.click_frequency = metrics.get('click_frequency')
        behavioral_data.typing_rhythm_consistency = metrics.get('typing_rhythm_consistency')
        
        # Save to database
        db.session.add(behavioral_data)
        db.session.commit()
        
        return jsonify({'status': 'success', 'data_id': behavioral_data.id})
    
    except Exception as e:
        logging.error(f"Error collecting behavioral data: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/detect_bot', methods=['POST'])
def detect_bot():
    """Endpoint to perform bot detection analysis"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'sessionId' not in data:
            return jsonify({'error': 'Invalid data'}), 400
        
        session_id = data['sessionId']
        
        # Get recent behavioral data for this session
        behavioral_data = BehavioralData.query.filter_by(
            session_id=session_id
        ).order_by(BehavioralData.timestamp.desc()).first()
        
        if not behavioral_data:
            return jsonify({'error': 'No behavioral data found'}), 404
        
        # Extract features for ML model
        features = behavioral_analyzer.extract_features(behavioral_data)
        
        # Make prediction
        prediction, confidence = ml_model.predict(features)
        
        # Update behavioral data with prediction
        behavioral_data.is_human = (prediction == 'human')
        behavioral_data.confidence_score = confidence
        
        # Log the detection
        detection_log = DetectionLog(
            session_id=session_id,
            prediction=prediction,
            confidence=confidence,
            page_url=data.get('page_url'),
            action_type=data.get('action_type', 'research_test'),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        db.session.add(detection_log)
        db.session.commit()
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'is_human': prediction == 'human',
            'processing_time_ms': detection_log.processing_time_ms
        })
    
    except Exception as e:
        logging.error(f"Error in bot detection: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint to get detection statistics"""
    try:
        # Get statistics for the last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_detections = DetectionLog.query.filter(
            DetectionLog.timestamp >= yesterday
        ).all()
        
        total = len(recent_detections)
        human_count = sum(1 for d in recent_detections if d.prediction == 'human')
        bot_count = total - human_count
        
        # Hourly breakdown for charts
        hourly_data = {}
        for detection in recent_detections:
            hour = detection.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = {'human': 0, 'bot': 0}
            hourly_data[hour][detection.prediction] += 1
        
        return jsonify({
            'total_detections': total,
            'human_detections': human_count,
            'bot_detections': bot_count,
            'hourly_data': hourly_data
        })
    
    except Exception as e:
        logging.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/research_test', methods=['POST'])
def research_test():
    """Research testing endpoint for behavioral analysis"""
    try:
        test_data = request.get_json()
        session_id = test_data.get('sessionId')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Perform behavioral analysis
        detection_data = {
            'sessionId': session_id,
            'page_url': '/research',
            'action_type': 'behavioral_analysis'
        }
        
        # Get recent behavioral data
        behavioral_data = BehavioralData.query.filter_by(
            session_id=session_id
        ).order_by(BehavioralData.timestamp.desc()).first()
        
        if behavioral_data:
            features = behavioral_analyzer.extract_features(behavioral_data)
            prediction, confidence = ml_model.predict(features)
            
            # Log the detection
            detection_log = DetectionLog(
                session_id=session_id,
                prediction=prediction,
                confidence=confidence,
                page_url='/research',
                action_type='behavioral_analysis'
            )
            
            db.session.add(detection_log)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Behavioral analysis completed successfully',
                'analysis': {
                    'classification': prediction,
                    'confidence_score': confidence,
                    'is_human_behavior': prediction == 'human',
                    'behavioral_features': len(features[0]) if features else 0
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Insufficient behavioral data for analysis'
            }), 400
    
    except Exception as e:
        logging.error(f"Error in research test: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
