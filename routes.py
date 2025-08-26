from flask import render_template, request, jsonify, session, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from app import app, db
from models import BehavioralData, DetectionLog, ModelMetrics, User, Task
from behavioral_analyzer import BehavioralAnalyzer
from ml_model import MLModel
import uuid
import time
from datetime import datetime, timedelta
import logging
import random

# Initialize components
behavioral_analyzer = BehavioralAnalyzer()
ml_model = MLModel()

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form

        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        # Validate input
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        if request.is_json:
            return jsonify({'success': True, 'message': 'Registration successful'})
        else:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if user.is_blocked:
                return jsonify({'error': 'Your account has been blocked due to suspicious behavior'}), 403

            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()

            if request.is_json:
                redirect_url = url_for('admin_dashboard') if user.is_admin else url_for('user_dashboard')
                return jsonify({'success': True, 'redirect': redirect_url})
            else:
                return redirect(url_for('admin_dashboard') if user.is_admin else url_for('user_dashboard'))

        return jsonify({'error': 'Invalid username or password'}), 401

    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def user_dashboard():
    """User dashboard with behavioral tasks"""
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))

    # Get user's tasks
    user_tasks = Task.query.filter_by(user_id=current_user.id).order_by(Task.created_at.desc()).all()

    # Create new tasks if user has completed all or has none
    if len([t for t in user_tasks if t.status == 'pending']) == 0:
        create_user_tasks(current_user.id)
        user_tasks = Task.query.filter_by(user_id=current_user.id).order_by(Task.created_at.desc()).all()

    # Get user's behavioral statistics
    recent_detections = DetectionLog.query.filter_by(
        session_id=session.get('session_id', '')
    ).order_by(DetectionLog.timestamp.desc()).limit(5).all()

    return render_template('dashboard/user_dashboard.html', 
                         tasks=user_tasks, 
                         recent_detections=recent_detections)

@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard for user management and analytics"""
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))

    # Get all users with their behavioral statistics
    users = User.query.filter_by(is_admin=False).order_by(User.created_at.desc()).all()

    # Recent detections
    recent_detections = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).limit(100).all()

    # Overall statistics
    total_users = len(users)
    active_users = len([u for u in users if u.last_login and u.last_login > datetime.utcnow() - timedelta(days=7)])
    blocked_users = len([u for u in users if u.is_blocked])
    suspected_bots = len([u for u in users if u.is_likely_bot])

    # Model metrics
    latest_metrics = ModelMetrics.query.order_by(ModelMetrics.timestamp.desc()).first()

    stats = {
        'total_users': total_users,
        'active_users': active_users,
        'blocked_users': blocked_users,
        'suspected_bots': suspected_bots,
        'total_detections': len(recent_detections),
        'human_detections': len([d for d in recent_detections if d.prediction == 'human']),
        'bot_detections': len([d for d in recent_detections if d.prediction == 'bot'])
    }

    return render_template('dashboard/admin_dashboard.html',
                         users=users,
                         stats=stats,
                         recent_detections=recent_detections[:20],
                         model_metrics=latest_metrics)

@app.route('/admin/block_user/<int:user_id>', methods=['POST'])
@login_required
def block_user(user_id):
    """Block a user"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    user = User.query.get_or_404(user_id)
    user.is_blocked = True
    db.session.commit()

    return jsonify({'success': True, 'message': f'User {user.username} has been blocked'})

@app.route('/admin/unblock_user/<int:user_id>', methods=['POST'])
@login_required
def unblock_user(user_id):
    """Unblock a user"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    user = User.query.get_or_404(user_id)
    user.is_blocked = False
    db.session.commit()

    return jsonify({'success': True, 'message': f'User {user.username} has been unblocked'})

@app.route('/task/<int:task_id>')
@login_required
def perform_task(task_id):
    """Task performance page"""
    task = Task.query.get_or_404(task_id)

    if task.user_id != current_user.id:
        return redirect(url_for('user_dashboard'))

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    return render_template('tasks/task_interface.html', task=task, session_id=session['session_id'])

@app.route('/research')
@login_required
def research():
    """Research interface for behavioral biometrics testing - Admin only"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('research.html', session_id=session['session_id'])

@app.route('/methodology')
def methodology():
    """Research methodology and technical details"""
    return render_template('methodology.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/analytics')
@login_required
def analytics():
    """Analytics dashboard - Admin only"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))

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

# API Endpoints

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
        logging.info(f"Received detection request: {data}")

        if not data or 'sessionId' not in data:
            return jsonify({'error': 'Invalid data - sessionId required'}), 400

        session_id = data['sessionId']
        task_id = data.get('taskId')

        # Get behavioral data from the current request ONLY (for task completion detection)
        request_mouse = data.get('mouseMovements', [])
        request_clicks = data.get('clickPatterns', [])
        request_keys = data.get('keystrokePatterns', [])
        request_scrolls = data.get('scrollPatterns', [])

        # For task completion detection, ONLY use current task data
        # This prevents false positives from accumulated session data
        current_task_clicks = len(request_clicks)
        current_task_mouse = len(request_mouse)
        current_task_keyboard = len(request_keys)
        current_task_scrolls = len(request_scrolls)

        # Still save to database for historical tracking
        behavioral_data = BehavioralData.query.filter_by(
            session_id=session_id
        ).order_by(BehavioralData.timestamp.desc()).first()

        # Combine all available data sources for database storage
        all_mouse_events = request_mouse[:]
        all_click_events = request_clicks[:]
        all_keyboard_events = request_keys[:]
        all_scroll_events = request_scrolls[:]

        if behavioral_data:
            # Add existing data from database for storage
            if behavioral_data.mouse_movements:
                all_mouse_events.extend(behavioral_data.mouse_movements)
            if behavioral_data.click_patterns:
                all_click_events.extend(behavioral_data.click_patterns)
            if behavioral_data.keystroke_patterns:
                all_keyboard_events.extend(behavioral_data.keystroke_patterns)
            if behavioral_data.scroll_patterns:
                all_scroll_events.extend(behavioral_data.scroll_patterns)

        # Create or update behavioral data record for storage
        if not behavioral_data:
            behavioral_data = BehavioralData(
                session_id=session_id,
                mouse_movements=all_mouse_events,
                click_patterns=all_click_events,
                scroll_patterns=all_scroll_events,
                keystroke_patterns=all_keyboard_events,
                user_agent=request.headers.get('User-Agent'),
                ip_address=request.remote_addr
            )
            db.session.add(behavioral_data)
            db.session.flush()
        else:
            # Update with combined data
            behavioral_data.mouse_movements = all_mouse_events
            behavioral_data.click_patterns = all_click_events
            behavioral_data.keystroke_patterns = all_keyboard_events
            behavioral_data.scroll_patterns = all_scroll_events

        # Use ONLY current task data for detection (not accumulated data)
        mouse_events = current_task_mouse
        click_events = current_task_clicks
        keyboard_events = current_task_keyboard
        scroll_events = current_task_scrolls

        logging.info(f"Session ID: {session_id}")
        logging.info(f"CURRENT TASK Events - Mouse: {mouse_events}, Clicks: {click_events}, Keyboard: {keyboard_events}, Scrolls: {scroll_events}")
        logging.info(f"ACCUMULATED Events - Mouse: {len(all_mouse_events)}, Clicks: {len(all_click_events)}, Keyboard: {len(all_keyboard_events)}, Scrolls: {len(all_scroll_events)}")
        logging.info(f"Request data keys: {list(data.keys())}")

        # Simplified bot detection logic - ONLY uses current task clicks
        # BOT: If current task clicks <= 2 (insufficient interaction for task completion)
        # HUMAN: If current task clicks >= 3 (shows meaningful human interaction)

        # Primary bot detection based on CURRENT TASK clicks only
        if click_events <= 2:
            prediction = 'bot'
            confidence = 0.90
            reason = [f"insufficient_current_task_clicks({click_events}<=2)"]
        else:
            prediction = 'human'
            confidence = 0.85
            reason = [f"sufficient_current_task_clicks({click_events}>=3)"]

        logging.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Reason: {', '.join(reason)}")

        # Update behavioral data with prediction
        behavioral_data.is_human = (prediction == 'human')
        behavioral_data.confidence_score = confidence

        # Log the detection
        detection_log = DetectionLog(
            session_id=session_id,
            prediction=prediction,
            confidence=confidence,
            page_url=data.get('page_url', ''),
            action_type=data.get('action_type', 'task_completion'),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

        db.session.add(detection_log)

        # Update user behavioral statistics if logged in
        if current_user.is_authenticated and not current_user.is_admin:
            current_user.update_behavioral_stats(prediction, confidence)

        # Update task if provided (using current task data, not accumulated)
        if task_id and current_user.is_authenticated:
            task = Task.query.filter_by(id=task_id, user_id=current_user.id).first()
            if task:
                task.status = 'completed'
                task.completed_at = datetime.utcnow()
                task.completion_time_ms = detection_log.processing_time_ms
                task.behavioral_score = confidence
                task.mouse_events = current_task_mouse  # Current task only
                task.keyboard_events = current_task_keyboard  # Current task only

        db.session.commit()

        logging.info(f"Detection completed: {prediction} with {confidence:.2f} confidence")

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'is_human': prediction == 'human',
            'processing_time_ms': detection_log.processing_time_ms
        })

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error in bot detection: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

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

@app.route('/api/admin/users')
@login_required
def get_users_data():
    """Get users data for admin charts"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403

    users = User.query.filter_by(is_admin=False).all()

    # Prepare data for charts
    user_data = []
    for user in users:
        user_data.append({
            'id': user.id,
            'username': user.username,
            'total_sessions': user.total_sessions,
            'bot_percentage': user.bot_percentage,
            'avg_confidence': user.avg_confidence_score,
            'is_blocked': user.is_blocked,
            'last_login': user.last_login.isoformat() if user.last_login else None
        })

    return jsonify({'users': user_data})

def create_user_tasks(user_id):
    """Create behavioral analysis tasks for a user"""
    task_templates = [
        {
            'title': 'Form Interaction Test',
            'description': 'Complete a simple form with natural mouse and keyboard interactions',
            'task_type': 'form_fill'
        },
        {
            'title': 'Click Pattern Analysis',
            'description': 'Perform a series of clicks to analyze your clicking behavior',
            'task_type': 'click_sequence'
        },
        {
            'title': 'Typing Behavior Assessment',
            'description': 'Type a given text to analyze your keystroke dynamics',
            'task_type': 'typing_test'
        },
        {
            'title': 'Mouse Movement Tracking',
            'description': 'Navigate through interactive elements to capture mouse patterns',
            'task_type': 'mouse_tracking'
        }
    ]

    # Create 2-3 random tasks for the user
    selected_tasks = random.sample(task_templates, k=min(3, len(task_templates)))

    for template in selected_tasks:
        task = Task(
            user_id=user_id,
            title=template['title'],
            description=template['description'],
            task_type=template['task_type']
        )
        db.session.add(task)

    db.session.commit()