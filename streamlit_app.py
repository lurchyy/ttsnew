import torch
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models
from scipy.io.wavfile import write as write_wav
import streamlit as st
from pydub import AudioSegment
import io
import time
import tempfile
import os
import glob
import subprocess
import sys

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Bark Text-to-Speech Generator",
    layout="wide"
)

# Fix for ffmpeg - attempt to install and make it available
try:
    from pydub.utils import which
    
    # Only install if not found
    if which("ffmpeg") is None:
        st.warning("Installing ffmpeg dependencies - this may take a moment...")
        
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
            st.success("ffmpeg installed successfully!")
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
    st.error(f"Note: Could not install ffmpeg. Some audio processing features may be limited. Error: {str(e)}")

# 🎨 Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 🔄 Setup session state for variables that need to persist between reruns
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""
if 'current_voice' not in st.session_state:
    st.session_state.current_voice = "v2/en_speaker_1"
if 'streaming_enabled' not in st.session_state:
    st.session_state.streaming_enabled = True
if 'last_streaming_check' not in st.session_state:
    st.session_state.last_streaming_check = time.time()
if 'debounce_time' not in st.session_state:
    st.session_state.debounce_time = 2  # seconds
if 'processed_segments' not in st.session_state:
    st.session_state.processed_segments = {}  # To store processed text segments and their audio

# 🛠 Fix for Torch compatibility
@st.cache_resource
def setup_torch_compatibility():
    torch.serialization.add_safe_globals({
        "numpy.core.multiarray.scalar": np.ndarray
    })

    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

# 📦 Load Bark models
@st.cache_resource
def load_bark_models():
    with st.spinner("📦 Loading Bark models (this may take a minute)..."):
        preload_models()
    st.session_state.models_loaded = True
    return True

# 🎙️ Selected Voice Presets
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

# Split text into natural segments for streaming processing
def split_text_into_segments(text):
    # Split by sentences or punctuation
    delimiters = ['. ', '! ', '? ', '; ', ': ']
    segments = []
    current_pos = 0
    
    # First try to split by punctuation
    for delimiter in delimiters:
        while True:
            pos = text.find(delimiter, current_pos)
            if pos == -1:
                break
            
            # Include the delimiter in the segment
            segment = text[current_pos:pos + len(delimiter)]
            segments.append(segment)
            current_pos = pos + len(delimiter)
    
    # Add any remaining text
    if current_pos < len(text):
        segments.append(text[current_pos:])
    
    # If punctuation splitting didn't work, split by chunks
    if not segments:
        # Split by fixed length if no natural breaks were found
        chunk_size = 100  # characters
        for i in range(0, len(text), chunk_size):
            segments.append(text[i:i+chunk_size])
    
    return segments

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
    if len(current_text) < len(previous_text):
        # Reset processed segments
        st.session_state.processed_segments = {}
        return current_text
    
    # Check if the previous text is a prefix of the current text
    if current_text.startswith(previous_text):
        return current_text[len(previous_text):]
    
    # If the text changed in the middle, we need to process everything again
    # Reset processed segments
    st.session_state.processed_segments = {}
    return current_text

# Combine audio segments
def combine_audio_segments(segments):
    # If there's only one segment, return it
    if len(segments) == 1:
        return segments[0]
    
    combined = None
    
    for segment in segments:
        if segment is None:
            continue
            
        # Convert to AudioSegment if it's a numpy array
        if isinstance(segment, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                write_wav(temp_file.name, rate=SAMPLE_RATE, data=segment)
                segment_audio = AudioSegment.from_wav(temp_file.name)
        elif isinstance(segment, bytes):
            # If it's bytes (already processed)
            segment_buffer = io.BytesIO(segment)
            segment_audio = AudioSegment.from_wav(segment_buffer)
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

# ▶️ Generation function for a text segment
def generate_audio_segment(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    if not text.strip():
        return None
        
    # Process text for more natural speech with pauses
    processed_text = process_text_for_speech(text, add_long_pauses)
    
    # Generate raw Bark audio with adjusted parameters
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
            
            # ⏩ Adjust speed
            if speed != 1.0:
                new_frame_rate = int(sound.frame_rate * speed)
                sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
                sound = sound.set_frame_rate(SAMPLE_RATE)

            # 🎵 Adjust pitch
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
        # Return raw audio as NumPy array
        return audio_array

# ▶️ Main generation function
def generate_audio_from_text(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses, streaming=False):
    if st.session_state.is_processing:
        return False
    
    st.session_state.is_processing = True
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if streaming:
            # Find what's new in the text
            new_text = find_new_text(st.session_state.last_text, text)
            
            if new_text:
                status_text.text("⏳ Processing new text...")
                progress_bar.progress(10)
                
                # Process only the new text
                new_audio = generate_audio_segment(
                    new_text, 
                    voice_preset, 
                    pitch, 
                    speed, 
                    text_temp, 
                    waveform_temp, 
                    add_long_pauses
                )
                
                # Store the new segment
                segment_key = f"{voice_preset}_{text_temp}_{waveform_temp}_{pitch}_{speed}_{add_long_pauses}_{len(st.session_state.processed_segments)}"
                st.session_state.processed_segments[segment_key] = new_audio
                
                # Combine all segments
                progress_bar.progress(80)
                status_text.text("🔄 Combining audio segments...")
                combined_audio = combine_audio_segments(list(st.session_state.processed_segments.values()))
                
                # Update the current audio
                st.session_state.current_audio = combined_audio
                
                # Update the last processed text
                st.session_state.last_text = text
                
                progress_bar.progress(100)
                status_text.text("✅ Audio generation complete!")
                return True
            else:
                # No new text to process
                progress_bar.progress(100)
                status_text.text("✅ No new text to process.")
                return False
        else:
            # Process the entire text at once (non-streaming mode)
            status_text.text("⏳ Processing text...")
            progress_bar.progress(10)
            
            # Clear previous segments for new processing
            st.session_state.processed_segments = {}
            
            status_text.text("🔊 Generating audio...")
            progress_bar.progress(30)
            
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
            
            # Store the complete audio
            st.session_state.current_audio = audio_data
            
            # Update the last processed text
            st.session_state.last_text = text
            
            progress_bar.progress(100)
            status_text.text("✅ Audio generation complete!")
            return True
            
    except Exception as e:
        st.error(f"❌ Error generating audio: {str(e)}")
        return False
    finally:
        st.session_state.is_processing = False

# Apply voice style presets
def apply_voice_style(style):
    if style == 'default':
        return 0.7, 0.7, True
    elif style == 'natural':
        return 0.6, 0.5, True
    elif style == 'expressive':
        return 0.9, 0.8, True
    # Commented out styles
    # elif style == 'calm':
    #     return 0.4, 0.4, True
    # elif style == 'precise':
    #     return 0.3, 0.2, False
    return 0.7, 0.7, True  # Default fallback

# Function to check for streaming text changes
def check_streaming_changes(current_text, voice_preset, pitch, speed, text_temp, waveform_temp, add_pauses):
    """Check if text has changed and trigger audio generation in streaming mode"""
    if not st.session_state.streaming_enabled or st.session_state.is_processing:
        return
    
    # Get current time and check if we need to debounce
    now = time.time()
    time_since_last_check = now - st.session_state.last_streaming_check
    
    # Check if voice changed
    voice_changed = voice_preset != st.session_state.current_voice
    
    # Check if text changed significantly
    text_changed = (current_text != st.session_state.last_text and 
                   len(current_text) > 5 and  # Minimum content length
                   time_since_last_check > st.session_state.debounce_time)
    
    # Generate audio if conditions are met
    if (voice_changed or text_changed) and not st.session_state.is_processing:
        st.session_state.last_streaming_check = now
        
        # Create a placeholder for status message
        streaming_status = st.empty()
        
        if voice_changed:
            # If voice changed, we need to regenerate everything
            streaming_status.info(f"🎤 Voice changed - Regenerating audio...")
            st.session_state.processed_segments = {}  # Reset segments
            st.session_state.current_voice = voice_preset
        else:
            streaming_status.info(f"🔄 Text changed - Processing new content...")
        
        # Generate the audio with streaming enabled
        success = generate_audio_from_text(
            current_text,
            voice_preset,
            pitch,
            speed,
            text_temp,
            waveform_temp,
            add_pauses,
            streaming=True
        )
        
        if success:
            st.session_state.last_text = current_text
            streaming_status.empty()
            st.rerun()

# Main app function
def main():
    # Initialize Torch compatibility
    setup_torch_compatibility()
    
    # Display header
    st.markdown("<div class='main-header'>🔊 Bark Text-to-Speech Generator</div>", unsafe_allow_html=True)
    
    # Load models on first run
    if not st.session_state.models_loaded:
        load_bark_models()
    
    # Layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input with callback for streaming detection
        text_input = st.text_area(
            "Text to speak:",
            value="Hello! I'm speaking with a human-like voice and a selected Voice.",
            height=150,
            key="text_input"
        )
        
        # Streaming toggle
        streaming_enabled = st.toggle(
            "Enable Incremental Streaming", 
            value=st.session_state.streaming_enabled,
            help="When enabled, only new text additions will be processed and appended to existing audio"
        )
        
        # Update session state
        if streaming_enabled != st.session_state.streaming_enabled:
            st.session_state.streaming_enabled = streaming_enabled
            if streaming_enabled:
                st.success("🔄 Incremental streaming enabled - New text will be added to existing audio")
            else:
                st.info("⏸️ Streaming disabled - Use the Generate button to create audio")
                # Reset segments when disabling streaming
                st.session_state.processed_segments = {}
        
        st.markdown("<div class='sub-header'>Voice Settings</div>", unsafe_allow_html=True)
        
        # Voice selection
        voice_selection = st.selectbox(
            "Choose Voice:",
            options=list(Voice_presets.keys()),
            index=0,
            key="voice_selection"
        )
        
        voice_preset = Voice_presets[voice_selection]
        
        # Pitch and speed in a single row
        col1a, col1b = st.columns(2)
        with col1a:
            pitch = st.slider("Pitch (semitones):", min_value=-10, max_value=10, value=0, step=1)
        with col1b:
            speed = st.slider("Speed:", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
        
        # Style presets
        st.markdown("<div class='sub-header'>Voice Style Presets</div>", unsafe_allow_html=True)
        
        col1c, col1d = st.columns([3, 1])
        with col1c:
            voice_style = st.selectbox(
                "Voice Style:",
                options=["Default", "Natural", "Expressive"],
                index=0,
                key="voice_style"
            )
            
        # Extract parameters based on style
        text_temp, waveform_temp, add_pauses = apply_voice_style(voice_style.lower())
        
        with col1d:
            if st.button("Apply Style", use_container_width=True):
                st.session_state.text_temp = text_temp
                st.session_state.waveform_temp = waveform_temp
                st.session_state.add_pauses = add_pauses
                # When style changes, reset the processed segments
                st.session_state.processed_segments = {}
                st.toast(f"Applied {voice_style} style settings!")
                
    with col2:
        st.markdown("<div class='sub-header'>Advanced Parameters</div>", unsafe_allow_html=True)
        
        # Use session state values if set by Apply Style button, otherwise use sliders
        if 'text_temp' not in st.session_state:
            st.session_state.text_temp = 0.7
        if 'waveform_temp' not in st.session_state:
            st.session_state.waveform_temp = 0.7
        if 'add_pauses' not in st.session_state:
            st.session_state.add_pauses = True
            
        text_temp = st.slider(
            "Temperature:", 
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.text_temp,
            step=0.05,
            help="Lower = more deterministic, Higher = more varied/creative"
        )
        
        waveform_temp = st.slider(
            "Waveform Temp:", 
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.waveform_temp,
            step=0.05,
            help="Controls variability in the audio waveform"
        )
        
        add_pauses = st.checkbox(
            "Natural Pauses", 
            value=st.session_state.add_pauses,
            help="Add longer pauses at punctuation for more natural speech"
        )
        
        st.markdown("<div class='sub-header'>Generate Audio</div>", unsafe_allow_html=True)
        
        # Generate button
        if st.button("Generate Audio 🎤", type="primary", use_container_width=True, disabled=st.session_state.is_processing):
            if not text_input.strip():
                st.warning("Please enter some text to generate audio.")
            else:
                with st.spinner("Generating audio..."):
                    # Clear previous segments for full regeneration
                    st.session_state.processed_segments = {}
                    
                    success = generate_audio_from_text(
                        text_input, 
                        voice_preset, 
                        pitch, 
                        speed, 
                        text_temp, 
                        waveform_temp, 
                        add_pauses,
                        streaming=False
                    )
                    if success:
                        st.session_state.last_text = text_input
                        st.session_state.current_voice = voice_preset
                        st.rerun()
    
    # Add streaming check functionality
    check_streaming_changes(text_input, voice_preset, pitch, speed, text_temp, waveform_temp, add_pauses)
                    
    # Audio Output Section
    st.markdown("<div class='sub-header'>Audio Output</div>", unsafe_allow_html=True)
    
    # Display audio if available
    if st.session_state.current_audio is not None:
        audio_data = st.session_state.current_audio
        
        # Determine if it's raw numpy array or bytes
        if isinstance(audio_data, np.ndarray):
            # For numpy array
            st.audio(audio_data, sample_rate=SAMPLE_RATE)
            
            # Create download button for the audio
            with io.BytesIO() as buffer:
                write_wav(buffer, rate=SAMPLE_RATE, data=audio_data)
                buffer.seek(0)
                st.download_button(
                    label="Download Audio 💾",
                    data=buffer,
                    file_name=f"bark_voice_{voice_selection.replace(' ', '_')}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        else:
            # For bytes (already processed audio)
            st.audio(audio_data, format="audio/wav")
            
            # Download button for bytes
            st.download_button(
                label="Download Audio 💾",
                data=audio_data,
                file_name=f"bark_voice_{voice_selection.replace(' ', '_')}.wav",
                mime="audio/wav",
                use_container_width=True
            )
            
    # Credits and info
    st.markdown("---")
    st.markdown("Powered by [Bark](https://github.com/suno-ai/bark) text-to-speech model.")

# Run the app
if __name__ == "__main__":
    # Run cleanup first
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
        
    # Launch the app
    main() 