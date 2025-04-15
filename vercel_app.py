from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
import os
import glob
import sys
import time
import base64
import logging
import io

# Import Vercel-specific setup
import vercel_setup

# Import our modularized components
from models import device, load_bark_models, optimize_for_cpu, setup_torch_compatibility, Voice_presets, cleanup_memory
from audio_utils import setup_ffmpeg, generate_audio_from_text, apply_voice_style
from streaming import generate_audio_streaming, cleanup_old_sessions, create_streaming_session

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Allow cross-origin requests

# Initialize app
def initialize_app():
    # Setup torch compatibility
    setup_torch_compatibility()
    
    # Optimize memory usage for CPU mode
    optimize_for_cpu()
    
    # Check ffmpeg setup
    setup_ffmpeg()
    
    # Load models
    load_bark_models()
    
    # Cleanup any temporary files
    try:
        bark_files = glob.glob("bark_output_*.wav")
        if bark_files:
            for file in bark_files:
                try:
                    os.remove(file)
                except Exception:
                    pass
    except Exception:
        pass
        
    logger.info(f"App initialized. Running on {device} device.")

# Register initialization with Flask
with app.app_context():
    initialize_app()

# Routes
@app.route('/')
def index():
    return render_template('index.html', device=device)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        data = request.json
        
        # Extract parameters
        text = data.get('text', '')
        voice_preset = data.get('voice_preset')
        style = data.get('style', 'default').lower()
        pitch = int(data.get('pitch', 0))
        speed = float(data.get('speed', 1.0))
        text_temp = float(data.get('text_temp', 0.7))
        waveform_temp = float(data.get('waveform_temp', 0.7))
        add_pauses = data.get('add_pauses', True)
        
        # If voice_preset is a voice name, lookup the actual preset
        if voice_preset in Voice_presets:
            voice_preset = Voice_presets[voice_preset]
        # If no valid preset, use default
        if not voice_preset or voice_preset not in list(Voice_presets.values()):
            voice_preset = Voice_presets['Voice 1']
        
        # Generate audio
        audio_data = generate_audio_from_text(
            text,
            voice_preset,
            pitch,
            speed,
            text_temp,
            waveform_temp,
            add_pauses
        )
        
        if audio_data:
            from bark import SAMPLE_RATE
            # Create response with audio data
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'success': True,
                'audio_data': audio_b64,
                'format': 'wav',
                'sample_rate': SAMPLE_RATE,
                'params': {
                    'voice_preset': voice_preset,
                    'style': style,
                    'pitch': pitch,
                    'speed': speed,
                    'text_temp': text_temp,
                    'waveform_temp': waveform_temp,
                    'add_pauses': add_pauses
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate audio'
            }), 500
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/generate_streaming', methods=['POST'])
def api_generate_streaming():
    try:
        data = request.json
        
        # Extract parameters
        session_id = data.get('session_id')
        if not session_id:
            # Create a unique session ID if not provided
            session_id = create_streaming_session()
            
        text = data.get('text', '')
        voice_preset = data.get('voice_preset')
        style = data.get('style', 'default').lower()
        pitch = int(data.get('pitch', 0))
        speed = float(data.get('speed', 1.0))
        text_temp = float(data.get('text_temp', 0.7))
        waveform_temp = float(data.get('waveform_temp', 0.7))
        add_pauses = data.get('add_pauses', True)
        
        # If voice_preset is a voice name, lookup the actual preset
        if voice_preset in Voice_presets:
            voice_preset = Voice_presets[voice_preset]
        # If no valid preset, use default
        if not voice_preset or voice_preset not in list(Voice_presets.values()):
            voice_preset = Voice_presets['Voice 1']
        
        # Generate audio streaming
        audio_data = generate_audio_streaming(
            session_id,
            text,
            voice_preset,
            pitch,
            speed,
            text_temp,
            waveform_temp,
            add_pauses
        )
        
        if audio_data:
            from bark import SAMPLE_RATE
            # Create response with audio data
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'success': True,
                'session_id': session_id,
                'audio_data': audio_b64,
                'format': 'wav',
                'sample_rate': SAMPLE_RATE,
                'params': {
                    'voice_preset': voice_preset,
                    'style': style,
                    'pitch': pitch,
                    'speed': speed,
                    'text_temp': text_temp,
                    'waveform_temp': waveform_temp,
                    'add_pauses': add_pauses
                }
            })
        else:
            return jsonify({
                'success': True,  # Still success but no new audio
                'session_id': session_id,
                'audio_data': None,
                'message': 'No new audio to generate'
            })
    
    except Exception as e:
        logger.error(f"Error in streaming generation endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reset_streaming', methods=['POST'])
def reset_streaming():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        # Create a new session ID
        new_session_id = create_streaming_session()
        
        return jsonify({
            'success': True,
            'session_id': new_session_id,
            'message': 'Streaming session reset successfully'
        })
    
    except Exception as e:
        logger.error(f"Error resetting streaming session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/voices', methods=['GET'])
def get_voices():
    voices = []
    for name, preset_id in Voice_presets.items():
        voices.append({
            'id': preset_id,
            'name': name
        })
    
    return jsonify({
        'success': True,
        'voices': voices
    })

@app.route('/api/styles', methods=['GET'])
def get_styles():
    styles = [
        {'id': 'default', 'name': 'Default'},
        {'id': 'natural', 'name': 'Natural'},
        {'id': 'expressive', 'name': 'Expressive'}
    ]
    
    return jsonify({
        'success': True,
        'styles': styles
    })

@app.route('/api/style_info', methods=['GET'])
def get_style_info():
    style = request.args.get('style', 'default').lower()
    
    style_info = {
        'default': 'Default style with balanced parameters.',
        'natural': 'More natural-sounding with lower temperatures for a consistent voice.',
        'expressive': 'More expressive with higher temperatures for creative variation.'
    }
    
    return jsonify({
        'success': True,
        'info': style_info.get(style, 'No information available for this style.')
    })

@app.route('/api/apply_style', methods=['GET'])
def apply_style():
    style = request.args.get('style', 'default').lower()
    
    # Define style parameters
    style_params = {
        'default': {
            'pitch': 0,
            'speed': 1.0,
            'text_temp': 0.7,
            'waveform_temp': 0.7
        },
        'natural': {
            'pitch': 0,
            'speed': 1.0,
            'text_temp': 0.6,
            'waveform_temp': 0.5
        },
        'expressive': {
            'pitch': 0,
            'speed': 1.0,
            'text_temp': 0.9,
            'waveform_temp': 0.8
        }
    }
    
    # Get parameters for the requested style or use default
    parameters = style_params.get(style, style_params['default'])
    
    return jsonify({
        'success': True,
        'style': style,
        'parameters': parameters
    })

@app.route('/api/download', methods=['POST'])
def download_audio():
    try:
        data = request.json
        audio_b64 = data.get('audio')
        filename = data.get('filename', 'bark_audio.wav')
        
        if not audio_b64:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_data = base64.b64decode(audio_b64)
        
        # Create in-memory file
        audio_io = io.BytesIO(audio_data)
        audio_io.seek(0)
        
        # Send file as attachment
        return send_file(
            audio_io,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        logger.error(f"Error in download endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Run session cleanup
    cleanup_old_sessions()
    
    return jsonify({
        'status': 'healthy',
        'device': device,
        'active_sessions': 0  # Use actual count when sessions implemented
    })
        
# Vercel needs this route for health checks
@app.route('/_vercel/insights/view', methods=['POST'])
def vercel_insights():
    return jsonify({'success': True})

@app.route('/_vercel/insights/session', methods=['POST']) 
def vercel_session():
    return jsonify({'success': True})

# Main entrypoint - not used by Vercel but useful for local testing
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 