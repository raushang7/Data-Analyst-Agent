#!/bin/bash

# Data Analyst Agent - Run Script

echo "Starting Data Analyst Agent..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export FLASK_APP=app.py
export FLASK_ENV=development

# Default PORT to 3000 if not set
export PORT=${PORT:-3000}

# Start the application
echo "Starting Flask application on port ${PORT}..."
python app.py