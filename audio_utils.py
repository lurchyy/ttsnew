import torch
import numpy as np
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
import io
import tempfile
import logging
from pydub import AudioSegment
import subprocess
import sys
import os
from models import device, cleanup_memory

logger = logging.getLogger(__name__)

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

# Find new text to process in streaming mode
def find_new_text(previous_text, current_text):
    if not previous_text:
        return current_text
    
    # If text got shorter, start over
    if len(current_text) < len(previous_text) or not current_text.startswith(previous_text):
        return current_text
    
    # Extract only the new characters
    return current_text[len(previous_text):]

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