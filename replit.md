# StealthCAPTCHA

## Overview

StealthCAPTCHA is an invisible bot detection system that uses advanced behavioral analysis and machine learning to distinguish between human users and bots without traditional CAPTCHA challenges. The system continuously monitors user interactions including mouse movements, click patterns, keystroke dynamics, and scroll behavior to provide real-time bot detection with minimal user friction.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
The application is built on Flask with SQLAlchemy ORM for database operations. The core architecture follows a modular design pattern with separated concerns:

- **Flask Application Core**: Main application setup with proxy fix middleware for deployment behind reverse proxies
- **Database Layer**: PostgreSQL database with SQLAlchemy models for behavioral data storage, detection logs, and model metrics
- **Machine Learning Pipeline**: Random Forest classifier with scikit-learn for behavioral pattern classification
- **Behavioral Analysis Engine**: Custom analyzer that processes user interaction data and extracts behavioral features

### Data Collection and Analysis
The system implements client-side JavaScript tracking that captures:
- Mouse movement patterns (velocity, trajectory, pauses)
- Click behavior (timing, spatial distribution, rhythm)
- Keystroke dynamics (dwell times, flight times, typing patterns)
- Scroll patterns (speed variance, direction consistency)
- Device fingerprinting (user agent, screen resolution, timezone, platform)

### Machine Learning Model
Uses a Random Forest classifier with the following design decisions:
- 100 estimators for robust prediction accuracy
- Balanced class weights to handle imbalanced datasets
- StandardScaler for feature normalization
- Model persistence with joblib for production deployment
- Continuous retraining capability with new behavioral data

### Security and Privacy
- Session-based tracking with UUID generation
- IP address logging for network analysis
- Configurable data retention policies
- No personally identifiable information collection beyond behavioral patterns

### Frontend Architecture
Multi-page web application with:
- Bootstrap 5 for responsive UI components
- Chart.js for real-time analytics visualization
- Custom CSS with CSS variables for consistent theming
- Modular JavaScript architecture for behavioral tracking

### API Design
RESTful endpoints for:
- Behavioral data submission (`/api/behavioral_data`)
- Real-time statistics (`/api/stats`)
- Administrative dashboard data (`/api/admin`)
- Model performance metrics

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web application framework with SQLAlchemy extension
- **PostgreSQL**: Primary database for behavioral data storage
- **scikit-learn**: Machine learning library for Random Forest classification
- **NumPy/Pandas**: Data processing and numerical computation

### Frontend Libraries
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome 6**: Icon library for UI elements
- **Chart.js**: JavaScript charting library for analytics visualization

### Production Infrastructure
- **Werkzeug ProxyFix**: Middleware for deployment behind reverse proxies
- **Database Connection Pooling**: Configured with pool recycling and pre-ping for reliability
- **Session Management**: Flask session handling with configurable secret keys

### Development and Deployment
- **Joblib**: Model serialization and persistence
- **Logging**: Python logging module for application monitoring
- **Environment Variables**: Configuration management for database URLs and session secrets

The system is designed for scalability with database connection pooling, efficient data collection intervals, and modular components that can be independently scaled or modified.