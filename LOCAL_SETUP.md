
# Local Development Setup with XAMPP

This guide helps you set up StealthCAPTCHA on your local machine using XAMPP.

## Prerequisites

1. **XAMPP installed** with MySQL running
2. **Python 3.11+** installed
3. **Git** (optional, for version control)

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or if using pyproject.toml
pip install -e .
```

### 2. Start XAMPP MySQL

- Open XAMPP Control Panel
- Start MySQL service
- Ensure it's running on port 3306

### 3. Setup Database

Run the setup script:

```bash
python setup_local_db.py
```

This will:
- Create `stealth_captcha` database
- Create all required tables
- Set up admin user (username: admin, password: admin)

### 4. Configure Environment

Set the database URL for local development:

**Windows (Command Prompt):**
```cmd
set DATABASE_URL=mysql+pymysql://root:@localhost/stealth_captcha
```

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL="mysql+pymysql://root:@localhost/stealth_captcha"
```

**Linux/Mac:**
```bash
export DATABASE_URL="mysql+pymysql://root:@localhost/stealth_captcha"
```

### 5. Run the Application

```bash
python main.py
```

The application will be available at: `http://localhost:5000`

## Default Login Credentials

- **Admin Access:**
  - Username: `admin`
  - Password: `admin`

## Database Configuration

The application automatically detects the database type:
- **Local:** Uses MySQL with XAMPP (`mysql+pymysql://root:@localhost/stealth_captcha`)
- **Replit:** Uses PostgreSQL (from `DATABASE_URL` environment variable)

## Troubleshooting

### Common Issues:

1. **"Access denied for user 'root'"**
   - Check if XAMPP MySQL is running
   - Verify root user has no password (default XAMPP setup)

2. **"Can't connect to MySQL server"**
   - Ensure MySQL is running in XAMPP
   - Check if port 3306 is available

3. **"Database doesn't exist"**
   - Run `python setup_local_db.py` again

4. **"Module not found" errors**
   - Install dependencies: `pip install pymysql flask-sqlalchemy`

### Reset Database:

If you need to reset the database:

1. Open phpMyAdmin (http://localhost/phpmyadmin)
2. Drop the `stealth_captcha` database
3. Run `python setup_local_db.py` again

## Project Structure

```
stealth_captcha/
├── app.py              # Main Flask application
├── models.py           # Database models
├── routes.py           # Application routes
├── behavioral_analyzer.py  # Behavioral analysis logic
├── ml_model.py         # Machine learning model
├── setup_local_db.py   # Database setup script
├── templates/          # HTML templates
├── static/            # CSS, JS files
└── LOCAL_SETUP.md     # This file
```

## Features Available Locally

- ✅ User registration and login
- ✅ Admin dashboard with user management
- ✅ Behavioral data collection
- ✅ Bot detection using ML
- ✅ Task-based behavioral analysis
- ✅ Real-time analytics and charts
- ✅ User blocking/unblocking functionality

Happy coding! 🚀
