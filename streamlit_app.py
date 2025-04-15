import streamlit as st
import torch
import numpy as np
import os
import io
import time
import base64
import uuid
import glob
import logging
import tempfile
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables 
models_loaded = False
streaming_sessions = {}

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Bark TTS Generator",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Detect CUDA availability and set device accordingly
def setup_device():
    if torch.cuda.is_available():
        try:
            # Test CUDA with a small tensor operation
            test_tensor = torch.zeros(1).cuda()
            test_tensor = test_tensor + 1
            device = "cuda"
            st.sidebar.success("✅ GPU detected! Using CUDA for faster processing.")
        except Exception as e:
            device = "cpu"
            st.sidebar.warning(f"⚠️ GPU detected but not working properly. Using CPU. Error: {str(e)}")
    else:
        device = "cpu"
        st.sidebar.warning("⚠️ No GPU detected. Using CPU (this will be slower).")
    
    # Set default device for torch
    torch.set_default_device(device)
    
    # If using CPU, optimize thread usage
    if device == "cpu":
        torch.set_num_threads(max(4, os.cpu_count() or 4))
        logger.info(f"CPU mode: Using {torch.get_num_threads()} CPU threads")
    
    return device

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
@st.cache_resource
def load_bark_models():
    with st.spinner("Loading Bark models... This may take a minute."):
        preload_models()
    st.success("Models loaded successfully!")
    return True

# Check and setup ffmpeg
def setup_ffmpeg():
    try:
        from pydub.utils import which
        
        # Only install if not found
        if which("ffmpeg") is None:
            st.warning("FFmpeg not found, trying to install...")
            
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
                st.success("FFmpeg installed successfully!")
            except:
                # Try installing a Python package as fallback
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "ffmpeg-python"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                st.info("Installed ffmpeg-python package as a fallback.")
        
        # Set environment variable to help pydub find ffmpeg
        os.environ["PATH"] += os.pathsep + os.path.dirname(which("ffmpeg") or "")
        
    except Exception as e:
        st.error(f"Could not install ffmpeg. Some audio processing features may be limited. Error: {str(e)}")

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

# Function to clean up memory (especially important for CPU mode)
def cleanup_memory():
    if device == "cpu":
        # Force garbage collection
        import gc
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

# Find new text to process in streaming mode
def find_new_text(previous_text, current_text):
    if not previous_text:
        return current_text
    
    # If text got shorter, start over
    if len(current_text) < len(previous_text) or not current_text.startswith(previous_text):
        return current_text
    
    # Extract only the new characters
    return current_text[len(previous_text):]

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

# Generate audio with streaming (incremental updates)
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
            # No new text, return whatever we have
            if not session["segments"]:
                return None
                
            combined_audio = combine_audio_segments(session["segments"])
            return combined_audio
        
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
        st.error(f"Error generating streaming audio: {str(e)}")
        return None

# Reset streaming session
def reset_streaming_session():
    session_id = str(uuid.uuid4())
    if 'streaming_session_id' in st.session_state:
        # Remove old session if it exists
        old_session_id = st.session_state.streaming_session_id
        if old_session_id in streaming_sessions:
            del streaming_sessions[old_session_id]
    
    st.session_state.streaming_session_id = session_id
    
    return session_id

# Main function
def main():
    # Initialize
    global device
    device = setup_device()
    setup_torch_compatibility()
    setup_ffmpeg()
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
    
    # Initialize session state for streaming
    if 'streaming_session_id' not in st.session_state:
        st.session_state.streaming_session_id = str(uuid.uuid4())
    
    # Main UI
    st.title("🔊 Bark Text-to-Speech Generator")
    
    # Device info
    st.markdown(f"Running on **{device.upper()}**")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input section
        st.subheader("Text Input")
        
        # Streaming toggle
        streaming_enabled = st.checkbox("Enable streaming mode", value=True, 
                                        help="Generate audio incrementally as you type")
        
        # Reset streaming button (only show if streaming is enabled)
        if streaming_enabled:
            if st.button("Reset Streaming Session"):
                reset_streaming_session()
                st.success("Streaming session reset successfully")
        
        # Text input
        text_input = st.text_area("Enter text to convert to speech", 
                                 height=200, 
                                 placeholder="Type your text here...")
        
        # Generate button
        generate_button = st.button("Generate Audio", type="primary")
        
        # Audio output section
        st.subheader("Audio Output")
        
        # Initialize audio placeholder
        audio_placeholder = st.empty()
        
        # Download button placeholder
        download_placeholder = st.empty()
    
    with col2:
        # Voice settings
        st.subheader("Voice Settings")
        
        # Voice selection
        voice_option = st.selectbox("Voice", options=list(Voice_presets.keys()))
        voice_preset = Voice_presets[voice_option]
        
        # Style presets
        style_options = ["Default", "Natural", "Expressive"]
        style = st.selectbox("Style", options=style_options).lower()
        
        # Apply style parameters
        if style == "default":
            default_text_temp = 0.7
            default_waveform_temp = 0.7
        elif style == "natural":
            default_text_temp = 0.6
            default_waveform_temp = 0.5
        elif style == "expressive":
            default_text_temp = 0.9
            default_waveform_temp = 0.8
        
        # Style info
        style_info = {
            "default": "Default style with balanced parameters.",
            "natural": "More natural-sounding with lower temperatures for a consistent voice.",
            "expressive": "More expressive with higher temperatures for creative variation."
        }
        
        st.info(style_info[style])
        
        # Advanced parameters
        st.subheader("Advanced Parameters")
        
        pitch = st.slider("Pitch adjustment", -12, 12, 0, 1, 
                         help="Adjust pitch in semitones (0 = no change)")
        
        speed = st.slider("Speed", 0.5, 1.5, 1.0, 0.1,
                         help="Adjust speaking speed (1.0 = normal speed)")
        
        text_temp = st.slider("Text temperature", 0.1, 1.0, default_text_temp, 0.1,
                             help="Higher = more creative/variable, Lower = more consistent")
        
        waveform_temp = st.slider("Waveform temperature", 0.1, 1.0, default_waveform_temp, 0.1,
                                 help="Higher = more creative/variable, Lower = more consistent")
        
        add_long_pauses = st.checkbox("Add long pauses", value=True,
                                    help="Add longer pauses after punctuation for more natural speech")
    
    # Process text
    if generate_button and text_input:
        with st.spinner("Generating audio..."):
            try:
                if streaming_enabled:
                    # Use streaming mode
                    session_id = st.session_state.streaming_session_id
                    audio_data = generate_audio_streaming(
                        session_id,
                        text_input,
                        voice_preset,
                        pitch,
                        speed,
                        text_temp,
                        waveform_temp,
                        add_long_pauses
                    )
                else:
                    # Use standard mode
                    audio_data = generate_audio_segment(
                        text_input,
                        voice_preset,
                        pitch,
                        speed,
                        text_temp,
                        waveform_temp,
                        add_long_pauses
                    )
                
                if audio_data:
                    # Display audio
                    audio_placeholder.audio(audio_data, format="audio/wav")
                    
                    # Enable download
                    download_placeholder.download_button(
                        "Download Audio",
                        data=audio_data,
                        file_name="generated_audio.wav",
                        mime="audio/wav"
                    )
                    
                    st.success("Audio generated successfully!")
                else:
                    st.error("Failed to generate audio. Please try again.")
            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")
    
    # Handle automatic streaming generation (if enabled and no button press needed)
    elif streaming_enabled and text_input and len(text_input) > 10:
        # Only attempt streaming if we have a reasonable amount of text
        # and if the text has changed since last check
        if 'last_streamed_text' not in st.session_state:
            st.session_state.last_streamed_text = ""
        
        if text_input != st.session_state.last_streamed_text:
            # Update with the latest text
            st.session_state.last_streamed_text = text_input
            
            # Process in streaming mode
            session_id = st.session_state.streaming_session_id
            audio_data = generate_audio_streaming(
                session_id,
                text_input,
                voice_preset,
                pitch,
                speed,
                text_temp,
                waveform_temp,
                add_long_pauses
            )
            
            if audio_data:
                # Display audio
                audio_placeholder.audio(audio_data, format="audio/wav")
                
                # Enable download
                download_placeholder.download_button(
                    "Download Audio",
                    data=audio_data,
                    file_name="generated_audio.wav",
                    mime="audio/wav"
                )

if __name__ == "__main__":
    main() 