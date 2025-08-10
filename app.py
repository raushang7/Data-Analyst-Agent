from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import tempfile
from werkzeug.utils import secure_filename
from agent.agent import DataAnalystAgent
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the agent
agent = DataAnalystAgent()

@app.route('/', methods=['GET'])
def home():
    """Home page with API information"""
    return jsonify({
        'service': 'Data Analyst Agent',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'api': '/api/ (POST)',
        },
        'description': 'AI-powered data analysis API that can source, prepare, analyze, and visualize any data.',
        'usage': 'Send POST request to /api/ with questions.txt file and optional data files',
        'example': 'curl -X POST "http://localhost:3000/api/" -F "questions.txt=@your_questions.txt"'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'data-analyst-agent'
    })

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    try:
        start_time = datetime.utcnow()
        logger.info(f"Received analysis request at {start_time}")
        
        # Check if files are present
        if not request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Process uploaded files
        uploaded_files = {}
        questions_file = None
        
        # Priority 1: Check for form key 'questions.txt' 
        if 'questions.txt' in request.files:
            file = request.files['questions.txt']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                questions_file = file_path
                logger.info(f"Found questions file via form key: {filename}")
        
        # Process other files
        for file_key in request.files:
            file = request.files[file_key]
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Skip if already processed as questions file
            if file_key == 'questions.txt' and questions_file:
                continue
                
            file.save(file_path)
            
            # Fallback: detect questions file by filename if not found via form key
            if not questions_file and ('question' in filename.lower()):
                questions_file = file_path
                logger.info(f"Found questions file via filename fallback: {filename}")
            else:
                uploaded_files[file_key] = file_path
            
            logger.info(f"Saved file: {filename}")
        
        # Ensure questions file exists
        if not questions_file:
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        # Read questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_text = f.read()
        
        logger.info(f"Processing questions: {questions_text[:200]}...")
        
        # Process with agent
        try:
            result = agent.process_request(questions_text, uploaded_files)
            
            # Clean up uploaded files
            for file_path in [questions_file] + list(uploaded_files.values()):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Request processed successfully in {processing_time:.2f} seconds")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Agent processing error: {str(e)}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': f'Request failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Data Analyst Agent on port {port}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

