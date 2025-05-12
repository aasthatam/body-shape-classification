#!/usr/bin/env python
"""
Helper script to run the Body Shape API from the project root directory.
This ensures all imports work correctly.
"""

import os
import sys

# Make sure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the Flask app
from api.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Body Shape API on port {port}...")
    print(f"Access the API at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 