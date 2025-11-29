#!/usr/bin/env python3
"""
Database setup script for local and production use.

Usage:
  - Locally (XAMPP): set ALLOW_DB_CREATE=1 and DATABASE_* env vars (or rely on defaults),
    then run: python setup_local_db.py
  - On Render (production): set RUN_INITIAL_TRAINING=0 (default) and ensure your DB already exists.
"""

import os
import sys
import pymysql
from sqlalchemy import create_engine
from app import app
from extensions import db

# Read DB config from env vars (fallbacks for local dev/XAMPP)
DB_USER = os.getenv("DATABASE_USER", os.getenv("DB_USER", "root"))
DB_PASS = os.getenv("DATABASE_PASSWORD", os.getenv("DB_PASS", ""))
DB_HOST = os.getenv("DATABASE_HOST", os.getenv("DB_HOST", "localhost"))
DB_PORT = os.getenv("DATABASE_PORT", os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DATABASE_NAME", os.getenv("DB_NAME", "stealth_captcha"))

# Control flags
ALLOW_DB_CREATE = os.getenv("ALLOW_DB_CREATE", "0") == "1"
RUN_INITIAL_TRAINING = os.getenv("RUN_INITIAL_TRAINING", "0") == "1"

def create_database_if_allowed():
    """
    Attempt to create the database if ALLOW_DB_CREATE==True and
    we are connecting to a host that looks local.
    """
    if not ALLOW_DB_CREATE:
        print("Database creation skipped (ALLOW_DB_CREATE != 1).")
        return True

    # Only allow creation on localhost to avoid trying to create DB on managed hosts
    if DB_HOST not in ("localhost", "127.0.0.1"):
        print(f"Refusing to create DB on remote host ({DB_HOST}). Set ALLOW_DB_CREATE=1 only for local dev.")
        return False

    try:
        print(f"Attempting to create database '{DB_NAME}' on local MySQL...")
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            charset='utf8mb4'
        )
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            print(f"✓ Database '{DB_NAME}' created (or already exists).")
        connection.commit()
        connection.close()
        return True
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

def setup_tables_and_admin():
    """Create tables via SQLAlchemy and ensure an admin user exists."""
    try:
        with app.app_context():
            # Import models to register them
            import models  # noqa: F401

            # Create all tables (safe if DB exists)
            db.create_all()
            print("✓ All tables created or already exist (db.create_all()).")

            # Create default admin user if missing
            from models import User
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                admin_user = User(
                    username='admin',
                    email='admin@stealthcaptcha.com',
                    is_admin=True
                )
                admin_user.set_password('admin')
                db.session.add(admin_user)
                db.session.commit()
                print("✓ Admin user created (username: admin, password: admin)")
            else:
                print("✓ Admin user already exists")
        return True
    except Exception as e:
        print(f"✗ Error setting up tables: {e}")
        return False

def run_optional_initial_training():
    """
    Run the ML initial training only if RUN_INITIAL_TRAINING==True.
    On Render, leave RUN_INITIAL_TRAINING=0 to avoid heavy CPU on deploy.
    """
    if not RUN_INITIAL_TRAINING:
        print("Initial ML training skipped (RUN_INITIAL_TRAINING != 1).")
        return True

    try:
        from ml_model import MLModel
        with app.app_context():
            print("Starting initial ML model training (this may take a while)...")
            ml_model_instance = MLModel()
            # If model exists and you still want to retrain, handle flags inside MLModel
            ml_model_instance.train_initial_model()
            print("✓ ML training finished (or model files exist).")
        return True
    except Exception as e:
        print(f"✗ Error during ML training step: {e}")
        return False

def main():
    print("Setup script starting...")
    # On production (remote DB), we assume DB already exists. Only attempt creation when allowed.
    if not create_database_if_allowed():
        print("Database creation failed or was refused. If you are on production, ensure the database exists and credentials are correct.")
        # Do not exit here; we may still try to create tables if DB exists.
        # sys.exit(1)

    # Ensure SQLALCHEMY_DATABASE_URI is set in your app config (app already imported)
    # We attempt to create tables; if DB doesn't exist this will fail.
    if not setup_tables_and_admin():
        print("Failed to setup tables. Check DB connectivity and credentials.")
        sys.exit(1)

    # Optional training (disabled by default)
    if not run_optional_initial_training():
        print("Initial training step failed (if requested).")
        sys.exit(1)

    print("\n🎉 Setup completed successfully.")
    print("If running locally with XAMPP, set:")
    print(f"  export DATABASE_USER={DB_USER}")
    print(f"  export DATABASE_PASSWORD={DB_PASS}")
    print(f"  export DATABASE_HOST={DB_HOST}")
    print(f"  export DATABASE_PORT={DB_PORT}")
    print(f"  export DATABASE_NAME={DB_NAME}")
    print("Then run: python main.py")

if __name__ == "__main__":
    main()
