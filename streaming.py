import time
import logging
import uuid
from audio_utils import find_new_text, generate_audio_segment, combine_audio_segments
from models import cleanup_memory, Voice_presets

logger = logging.getLogger(__name__)

# Session storage for streaming audio (in a real app, use a proper session store)
streaming_sessions = {}

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

# Create a new streaming session
def create_streaming_session():
    return str(uuid.uuid4()) 