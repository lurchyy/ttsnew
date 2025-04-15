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

# Page configuration
st.set_page_config(
    page_title="Bark Text-to-Speech Generator",
    layout="wide"
)

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

# ▶️ Generation function
def generate_audio_from_text(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    if st.session_state.is_processing:
        return
    
    st.session_state.is_processing = True
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("⏳ Processing text...")
        progress_bar.progress(10)
        
        # Process text for more natural speech with pauses
        processed_text = process_text_for_speech(text, add_long_pauses)
        
        status_text.text("🔊 Generating raw audio...")
        progress_bar.progress(30)
                
        # 🔊 Generate raw Bark audio with adjusted parameters
        audio_array = generate_audio(
            processed_text, 
            history_prompt=voice_preset,
            text_temp=text_temp,
            waveform_temp=waveform_temp
        )
        
        progress_bar.progress(70)
        st.session_state.current_audio = audio_array  # Store raw audio for possible download
        
        # Apply pitch and speed adjustments directly to the numpy array
        if pitch != 0 or speed != 1.0:
            status_text.text("✨ Applying voice adjustments...")
            
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
                
                # Update current_audio with adjusted version
                st.session_state.current_audio = buffer.read()
        
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
    elif style == 'calm':
        return 0.4, 0.4, True
    elif style == 'precise':
        return 0.3, 0.2, False
    return 0.7, 0.7, True  # Default fallback

# Main app
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
        # Text input
        text_input = st.text_area(
            "Text to speak:",
            value="Hello! I'm speaking with a human-like voice and a selected Voice.",
            height=150,
            key="text_input"
        )
        
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
                options=["Default", "Natural", "Expressive", "Calm", "Precise"],
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
                    success = generate_audio_from_text(
                        text_input, 
                        voice_preset, 
                        pitch, 
                        speed, 
                        text_temp, 
                        waveform_temp, 
                        add_pauses
                    )
                    if success:
                        st.session_state.last_text = text_input
                        st.session_state.current_voice = voice_preset
                        st.rerun()
                    
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
