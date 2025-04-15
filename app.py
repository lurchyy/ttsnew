import torch
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models
from scipy.io.wavfile import write as write_wav
import io
import os
import time
import tempfile
import glob
import subprocess
import sys
import gc
import logging
import base64
import json
import uuid
from pydub import AudioSegment
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from flask_cors import CORS
from werkzeug.serving import run_simple
from waitress import serve

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Allow cross-origin requests

# Global variables
models_loaded = False
current_voice = "v2/en_speaker_1"

# Session storage for streaming audio (in a real app, use a proper session store)
streaming_sessions = {}

# Detect CUDA availability and set device accordingly
device = "cpu"
if torch.cuda.is_available():
    try:
        # Test CUDA with a small tensor operation
        test_tensor = torch.zeros(1).cuda()
        test_tensor = test_tensor + 1
        device = "cuda"
        logger.info("GPU detected and working! Using CUDA for faster processing.")
    except Exception as e:
        logger.warning(f"GPU detected but not working properly. Falling back to CPU. Error: {str(e)}")
        device = "cpu"
else:
    logger.info("No GPU detected. Using CPU for processing (this will be slower).")

# Set default device for torch
torch.set_default_device(device)

# Fix for Torch compatibility
def setup_torch_compatibility():
    torch.serialization.add_safe_globals({
        "numpy.core.multiarray.scalar": np.ndarray
    })

    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

# Load Bark models
def load_bark_models():
    global models_loaded
    if not models_loaded:
        logger.info(f"Loading Bark models on {device.upper()}...")
        preload_models()
        models_loaded = True
        logger.info("Models loaded successfully!")
    return True

# Selected Voice Presets
Voice_presets = {
    "Voice 1": "v2/en_speaker_1",
    "Voice 2": "v2/en_speaker_2",
    "Voice 3": "v2/en_speaker_3",
    "Voice 4": "v2/en_speaker_4",
    "Voice 5": "v2/en_speaker_5",
    "Voice 6": "v2/en_speaker_6",
    "Voice 7": "v2/en_speaker_7",
    "Voice 8": "v2/en_speaker_8",
    "Voice 9": "v2/en_speaker_9",
    "British Voice": "v2/en_speaker_0",
}

# Find new text to process in streaming mode
def find_new_text(previous_text, current_text):
    if not previous_text:
        return current_text
    
    # If text got shorter, start over
    if len(current_text) < len(previous_text) or not current_text.startswith(previous_text):
        return current_text
    
    # Extract only the new characters
    return current_text[len(previous_text):]

# Modify text to add more natural pauses if needed
def process_text_for_speech(text, add_long_pauses=True):
    if not add_long_pauses:
        return text
    
    # Add longer pauses at punctuation
    text = text.replace('. ', '.  ')  # Double space after periods
    text = text.replace('! ', '!  ')  # Double space after exclamation marks
    text = text.replace('? ', '?  ')  # Double space after question marks
    text = text.replace(', ', ',  ')  # Double space after commas
    text = text.replace('; ', ';  ')  # Double space after semicolons
    text = text.replace(':", "', ':"  "')  # Double space after quotes
    
    return text

# Combine audio segments
def combine_audio_segments(segments):
    if not segments:
        return None
        
    # If there's only one segment, return it
    if len(segments) == 1:
        return segments[0]
    
    combined = None
    
    # Sort segments by their keys (which should be indices)
    sorted_segments = sorted(segments.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    
    for _, segment in sorted_segments:
        if segment is None:
            continue
            
        # Convert to AudioSegment
        if isinstance(segment, bytes):
            # If it's bytes (already processed)
            segment_buffer = io.BytesIO(segment)
            segment_audio = AudioSegment.from_wav(segment_buffer)
        elif isinstance(segment, np.ndarray):
            # If it's numpy array
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                write_wav(temp_file.name, rate=SAMPLE_RATE, data=segment)
                segment_audio = AudioSegment.from_wav(temp_file.name)
        else:
            # Skip if invalid format
            continue
            
        if combined is None:
            combined = segment_audio
        else:
            combined += segment_audio
    
    # Convert back to bytes
    if combined:
        buffer = io.BytesIO()
        combined.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()
    
    return None

# Function to clean up memory (especially important for CPU mode)
def cleanup_memory():
    if device == "cpu":
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if it was used at some point
        if hasattr(torch, 'cuda'):
            try:
                torch.cuda.empty_cache()
            except:
                pass
                
        # Clear any cached tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
        
        # Final garbage collection
        gc.collect()

# Apply voice style presets
def apply_voice_style(style):
    if style == 'default':
        return 0.7, 0.7, True
    elif style == 'natural':
        return 0.6, 0.5, True
    elif style == 'expressive':
        return 0.9, 0.8, True
    return 0.7, 0.7, True  # Default fallback

# Generate audio for a text segment
def generate_audio_segment(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    if not text.strip():
        return None
        
    # Process text for more natural speech with pauses
    processed_text = process_text_for_speech(text, add_long_pauses)
    
    # Generate raw Bark audio with adjusted parameters
    with torch.device(device):
        audio_array = generate_audio(
            processed_text, 
            history_prompt=voice_preset,
            text_temp=text_temp,
            waveform_temp=waveform_temp
        )
    
    # Apply pitch and speed adjustments directly to the numpy array
    if pitch != 0 or speed != 1.0:
        # Convert numpy array to AudioSegment for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Write audio data to temp file
            write_wav(temp_file.name, rate=SAMPLE_RATE, data=audio_array)
            
            # Apply adjustments
            sound = AudioSegment.from_wav(temp_file.name)
            
            # Adjust speed
            if speed != 1.0:
                new_frame_rate = int(sound.frame_rate * speed)
                sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
                sound = sound.set_frame_rate(SAMPLE_RATE)

            # Adjust pitch
            if pitch != 0:
                pitch_factor = 2 ** (pitch / 12)
                new_rate = int(sound.frame_rate * pitch_factor)
                sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_rate})
                sound = sound.set_frame_rate(SAMPLE_RATE)
            
            # Export to in-memory file
            buffer = io.BytesIO()
            sound.export(buffer, format="wav")
            buffer.seek(0)
            
            # Return processed audio as bytes
            return buffer.read()
    else:
        # Convert numpy array to bytes
        with io.BytesIO() as buffer:
            write_wav(buffer, rate=SAMPLE_RATE, data=audio_array)
            buffer.seek(0)
            return buffer.read()

# Main generation function for non-streaming mode
def generate_audio_from_text(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    try:
        logger.info(f"Generating audio for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Process entire text
        audio_data = generate_audio_segment(
            text, 
            voice_preset, 
            pitch, 
            speed, 
            text_temp, 
            waveform_temp, 
            add_long_pauses
        )
        
        # Clean up memory in CPU mode
        cleanup_memory()
        
        return audio_data
            
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

# Generate audio for streaming with session management
def generate_audio_streaming(session_id, text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    try:
        # Create session if it doesn't exist
        if session_id not in streaming_sessions:
            streaming_sessions[session_id] = {
                "last_text": "",
                "segments": {},
                "voice_preset": voice_preset,
                "params": {
                    "pitch": pitch,
                    "speed": speed,
                    "text_temp": text_temp,
                    "waveform_temp": waveform_temp,
                    "add_long_pauses": add_long_pauses
                },
                "last_access": time.time()
            }
            logger.info(f"Created new streaming session: {session_id}")
        
        session = streaming_sessions[session_id]
        session["last_access"] = time.time()
        
        # Check if voice or parameter settings changed
        params_changed = (
            session["voice_preset"] != voice_preset or
            session["params"]["pitch"] != pitch or
            session["params"]["speed"] != speed or
            session["params"]["text_temp"] != text_temp or
            session["params"]["waveform_temp"] != waveform_temp or
            session["params"]["add_long_pauses"] != add_long_pauses
        )
        
        # If parameters changed, reset the session
        if params_changed:
            logger.info(f"Voice or parameters changed, resetting session {session_id}")
            session["segments"] = {}
            session["last_text"] = ""
            session["voice_preset"] = voice_preset
            session["params"] = {
                "pitch": pitch,
                "speed": speed,
                "text_temp": text_temp,
                "waveform_temp": waveform_temp,
                "add_long_pauses": add_long_pauses
            }
        
        # Find new text
        new_text = find_new_text(session["last_text"], text)
        
        if not new_text:
            logger.info(f"No new text to process for session {session_id}")
            # No new text, return whatever we have
            if not session["segments"]:
                return None
                
            combined_audio = combine_audio_segments(session["segments"])
            return combined_audio
            
        logger.info(f"Processing new text for streaming: {new_text[:30]}{'...' if len(new_text) > 30 else ''}")
        
        # Process the new text
        new_audio = generate_audio_segment(
            new_text,
            voice_preset,
            pitch,
            speed,
            text_temp,
            waveform_temp,
            add_long_pauses
        )
        
        # Store the new segment with a unique index
        segment_key = str(len(session["segments"]))
        session["segments"][segment_key] = new_audio
        
        # Update the last processed text
        session["last_text"] = text
        
        # Combine all segments
        combined_audio = combine_audio_segments(session["segments"])
        
        # Clean up memory in CPU mode
        cleanup_memory()
        
        return combined_audio
        
    except Exception as e:
        logger.error(f"Error in streaming audio generation: {str(e)}")
        return None

# Check and setup ffmpeg
def setup_ffmpeg():
    try:
        from pydub.utils import which
        
        # Only install if not found
        if which("ffmpeg") is None:
            logger.warning("FFmpeg not found, trying to install...")
            
            # Try apt-get if available (Linux)
            try:
                subprocess.check_call(
                    ["apt-get", "update", "-y"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                subprocess.check_call(
                    ["apt-get", "install", "-y", "ffmpeg"],
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                logger.info("FFmpeg installed successfully!")
            except:
                # Try installing a Python package as fallback
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "ffmpeg-python"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("Installed ffmpeg-python package as a fallback.")
        
        # Set environment variable to help pydub find ffmpeg
        os.environ["PATH"] += os.pathsep + os.path.dirname(which("ffmpeg") or "")
        
    except Exception as e:
        logger.error(f"Could not install ffmpeg. Some audio processing features may be limited. Error: {str(e)}")

# Initialize app (using Flask 2.3+ compatible approach)
def initialize_app():
    # Setup torch compatibility
    setup_torch_compatibility()
    
    # Optimize memory usage for CPU mode
    if device == "cpu":
        # Apply memory optimizations when running on CPU
        torch.set_num_threads(max(4, os.cpu_count() or 4))  # Set reasonable thread count
        logger.info(f"CPU mode: Using {torch.get_num_threads()} CPU threads")
    
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

# Register initialization with Flask
with app.app_context():
    initialize_app()

# Session cleanup task (would need a proper scheduler in production)
def cleanup_old_sessions():
    # This is a simple cleanup that could be expanded in a real app
    current_time = time.time()
    to_remove = []
    
    for session_id, session in streaming_sessions.items():
        if 'last_access' in session and current_time - session['last_access'] > 3600:  # 1 hour timeout
            to_remove.append(session_id)
    
    for session_id in to_remove:
        logger.info(f"Removing expired session: {session_id}")
        del streaming_sessions[session_id]

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
            session_id = str(uuid.uuid4())
            
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
        
        if session_id and session_id in streaming_sessions:
            # Reset the session
            logger.info(f"Manually resetting streaming session: {session_id}")
            del streaming_sessions[session_id]
            return jsonify({
                'success': True,
                'message': 'Streaming session reset successfully'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No active session to reset'
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
        'models_loaded': models_loaded,
        'active_sessions': len(streaming_sessions)
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    
    # Setup for production
    if os.environ.get('FLASK_ENV') == 'production':
        # Use waitress for production
        logger.info(f"Starting production server on port {port}")
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        # Use development server
        logger.info(f"Starting development server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True) 