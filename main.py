# main.py
import os
from app import app

if __name__ == "__main__":
    # Use the port Render (or other hosts) provides, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    # Do NOT run debug mode in production
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
