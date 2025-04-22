import gradio as gr
import torch
import numpy as np
import os
import io
import time
import uuid
import glob
import logging
import tempfile
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
import subprocess
import sys
import gc
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables 
models_loaded = False
streaming_sessions = {}
device = "cpu"  # Default to CPU, will detect GPU if available

# Streaming constants
MIN_STREAMING_CHARS = 15   # Minimum amount of new text before processing
MIN_STREAMING_DELAY = 1.5  # Minimum seconds between streaming updates

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

# Styles with their default parameters
style_presets = {
    "Default": {
        "text_temp": 0.6,
        "waveform_temp": 0.6,
        "description": "Default style with balanced parameters."
    },
    "Natural": {
        "text_temp": 0.5,
        "waveform_temp": 0.4,
        "description": "More natural-sounding with lower temperatures for a consistent voice."
    },
    "Expressive": {
        "text_temp": 0.8,
        "waveform_temp": 0.7,
        "description": "More expressive with higher temperatures for creative variation."
    },
    "Clean Speech": {
        "text_temp": 0.4,
        "waveform_temp": 0.4,
        "description": "Minimizes artifacts and garbled speech with very low temperatures."
    }
}

# Session state for streaming mode
class SessionState:
    def __init__(self):
        self.streaming_sessions = {}
        self.current_session_id = str(uuid.uuid4())
        self.debounce_time = 1.0  # Increased from 0.8 to give more time between updates
        self.last_update_time = 0
        self.current_audio = None

state = SessionState()

# Detect CUDA availability and set device accordingly
def setup_device():
    global device
    if torch.cuda.is_available():
        try:
            # Test CUDA with a small tensor operation
            test_tensor = torch.zeros(1).cuda()
            test_tensor = test_tensor + 1
            device = "cuda"
            logger.info("GPU detected and working! Using CUDA for faster processing.")
            return "GPU detected! Using CUDA for faster processing."
        except Exception as e:
            device = "cpu"
            logger.warning(f"GPU detected but not working properly. Falling back to CPU. Error: {str(e)}")
            return "GPU detected but not working properly. Using CPU."
    else:
        device = "cpu"
        logger.info("No GPU detected. Using CPU for processing (this will be slower).")
        
        # If using CPU, optimize thread usage
        torch.set_num_threads(max(4, os.cpu_count() or 4))
        logger.info(f"CPU mode: Using {torch.get_num_threads()} CPU threads")
        
        return "No GPU detected. Using CPU (this will be slower)."

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
                return "FFmpeg installed successfully!"
            except:
                # Try installing a Python package as fallback
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "ffmpeg-python"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("Installed ffmpeg-python package as a fallback.")
                return "Installed ffmpeg-python package as a fallback."
        
        # Set environment variable to help pydub find ffmpeg
        os.environ["PATH"] += os.pathsep + os.path.dirname(which("ffmpeg") or "")
        return "FFmpeg is ready."
        
    except Exception as e:
        error_msg = f"Could not install ffmpeg. Some audio processing features may be limited. Error: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Modify text to add more natural pauses if needed
def process_text_for_speech(text, add_long_pauses=True):
    # Clean and normalize the text first
    text = text.strip()
    
    # Ensure text ends with proper punctuation to signal the model to stop
    if text and not text[-1] in ['.', '!', '?']:
        text = text + "."
    
    # Add proper spacing after punctuation
    text = re.sub(r'([.!?,;:])\s*', r'\1 ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    if not add_long_pauses:
        return text
    
    # Add longer pauses at punctuation
    text = text.replace('. ', '.  ')  # Double space after periods
    text = text.replace('! ', '!  ')  # Double space after exclamation marks
    text = text.replace('? ', '?  ')  # Double space after question marks
    text = text.replace(', ', ',  ')  # Double space after commas
    text = text.replace('; ', ';  ')  # Double space after semicolons
    
    return text

# Function to trim silence and potential garbage at the end of audio
def trim_audio_end(audio_array, sample_rate, silence_threshold=0.01, min_silence_duration=0.2):
    """Trim silence and potential garbage at the end of the audio"""
    try:
        # Convert to AudioSegment for easier processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            write_wav(temp_file.name, rate=sample_rate, data=audio_array)
            audio = AudioSegment.from_wav(temp_file.name)
            
        # Calculate silence threshold in dBFS
        silence_thresh = audio.dBFS + 10  # Adjust based on your needs
        
        # Trim silence from the end
        trimmed_audio = audio.reverse().strip_silence(
            silence_thresh=silence_thresh,
            silence_len=int(min_silence_duration * 1000),  # Convert to ms
            padding=50  # Add 50ms padding
        ).reverse()
        
        # Convert back to numpy array
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            trimmed_audio.export(temp_file.name, format="wav")
            _, trimmed_array = SAMPLE_RATE, np.array(AudioSegment.from_wav(temp_file.name).get_array_of_samples())
        
        return (sample_rate, trimmed_array)
    except Exception as e:
        logger.warning(f"Error trimming audio: {str(e)}")
        # Return original audio if trimming fails
        return (sample_rate, audio_array)

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

# Generate audio for a text segment
def generate_audio_segment(text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    try:
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
        
        # Apply pitch and speed adjustments if needed
        if pitch != 0 or speed != 1.0:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                write_wav(temp_file.name, rate=SAMPLE_RATE, data=audio_array)
                sound = AudioSegment.from_wav(temp_file.name)
                
                if speed != 1.0:
                    new_frame_rate = int(sound.frame_rate * speed)
                    sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
                    sound = sound.set_frame_rate(SAMPLE_RATE)
                
                if pitch != 0:
                    pitch_factor = 2 ** (pitch / 12)
                    new_rate = int(sound.frame_rate * pitch_factor)
                    sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_rate})
                    sound = sound.set_frame_rate(SAMPLE_RATE)
                
                # Convert back to numpy array
                buffer = io.BytesIO()
                sound.export(buffer, format="wav")
                buffer.seek(0)
                audio_array = np.array(AudioSegment.from_wav(buffer).get_array_of_samples())
        
        # Return as tuple of (sample_rate, audio_array)
        return (SAMPLE_RATE, audio_array)
        
    except Exception as e:
        logger.error(f"Error generating audio segment: {str(e)}")
        return None

# Find new text to process in streaming mode
def find_new_text(previous_text, current_text):
    if not previous_text:
        return current_text
    
    # If text got shorter, start over
    if len(current_text) < len(previous_text) or not current_text.startswith(previous_text):
        return current_text
    
    # Extract only the new characters
    return current_text[len(previous_text):]

# Generate audio with streaming (incremental updates)
def generate_audio_streaming(session_id, text, voice_preset, pitch, speed, text_temp, waveform_temp, add_long_pauses):
    try:
        # Access global state
        global state
        
        # Create session if it doesn't exist
        if session_id not in state.streaming_sessions:
            state.streaming_sessions[session_id] = {
                "last_text": "",
                "current_audio": None,
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
        
        session = state.streaming_sessions[session_id]
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
            session["current_audio"] = None
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
            # No new text, return current audio
            return session["current_audio"]
        
        # Process the new text - only if we have meaningful content
        if len(new_text.strip()) > 0:
            try:
                new_audio = generate_audio_segment(
                    text,  # Process the entire text instead of just new text
                    voice_preset,
                    pitch,
                    speed,
                    text_temp,
                    waveform_temp,
                    add_long_pauses
                )
                
                if new_audio:
                    # Store the new audio
                    session["current_audio"] = new_audio
                    session["last_text"] = text
            except Exception as segment_error:
                logger.error(f"Error processing segment: {str(segment_error)}")
                # Continue with existing audio
        
        # Clean up memory in CPU mode
        cleanup_memory()
        
        return session["current_audio"]
        
    except Exception as e:
        logger.error(f"Error in streaming audio generation: {str(e)}")
        return None

# Reset streaming session
def reset_streaming_session():
    global state
    state.current_session_id = str(uuid.uuid4())
    state.last_update_time = 0  # Reset the timer
    
    # Remove old sessions
    if state.current_session_id in state.streaming_sessions:
        del state.streaming_sessions[state.current_session_id]
        
    logger.info(f"Created new streaming session: {state.current_session_id}")
    return "Streaming session reset successfully!"

# Text input change handler for auto-streaming
def handle_text_change(text, voice, style, use_streaming, pitch, speed, text_temp, waveform_temp, add_pauses):
    if not use_streaming:
        return None, "Streaming disabled"
    
    # Get voice preset
    voice_preset = Voice_presets[voice]
    
    # Get style parameters if not explicitly provided
    style_preset = style_presets[style]
    if text_temp is None:
        text_temp = style_preset["text_temp"]
    if waveform_temp is None:
        waveform_temp = style_preset["waveform_temp"]
    
    try:
        # Generate audio directly
        audio_data = generate_audio_segment(
            text,
            voice_preset,
            pitch,
            speed,
            text_temp,
            waveform_temp,
            add_pauses
        )
        
        if audio_data and isinstance(audio_data, tuple) and len(audio_data) == 2:
            return audio_data, "Audio generated successfully"
        else:
            return None, "Error generating audio"
            
    except Exception as e:
        logger.error(f"Error in text change handler: {str(e)}")
        return None, f"Error: {str(e)}"

# Main function to generate audio
def generate_audio_func(
    text, 
    voice, 
    style, 
    use_streaming,
    pitch=0, 
    speed=1.0, 
    text_temp=None, 
    waveform_temp=None, 
    add_long_pauses=True,
    progress=gr.Progress()
):
    if not text.strip():
        return None, "Please enter some text to generate audio."
    
    # Get voice preset
    voice_preset = Voice_presets[voice]
    
    # Get style parameters if not explicitly provided
    style_preset = style_presets[style]
    if text_temp is None:
        text_temp = style_preset["text_temp"]
    if waveform_temp is None:
        waveform_temp = style_preset["waveform_temp"]
    
    try:
        progress(0.1, "Preparing...")
        
        if use_streaming:
            progress(0.3, "Setting up streaming...")
            
            # Use streaming mode
            audio_data = generate_audio_streaming(
                state.current_session_id,
                text,
                voice_preset,
                pitch,
                speed,
                text_temp,
                waveform_temp,
                add_long_pauses
            )
        else:
            progress(0.3, "Processing text...")
            progress(0.5, "Generating audio...")
            
            # Use standard mode
            audio_data = generate_audio_segment(
                text,
                voice_preset,
                pitch,
                speed,
                text_temp,
                waveform_temp,
                add_long_pauses
            )
        
        progress(0.8, "Processing audio...")
        
        if audio_data:
            progress(1.0, "Complete!")
            return audio_data, "Audio generated successfully!"
        else:
            return None, "Failed to generate audio. Please try again."
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None, f"Error: {str(e)}"

# Style change handler
def style_change(style):
    """Update UI values when style changes"""
    style_preset = style_presets[style]
    return style_preset["text_temp"], style_preset["waveform_temp"], style_preset["description"]

# Initialization function
def initialize():
    # Setup compatibility
    setup_torch_compatibility()
    
    # Setup ffmpeg
    ffmpeg_status = setup_ffmpeg()
    
    # Detect device
    device_status = setup_device()
    
    # Load models
    try:
        load_bark_models()
        model_status = "Models loaded successfully!"
    except Exception as e:
        model_status = f"Error loading models: {str(e)}"
    
    # Initialize streaming state
    global state
    state.last_update_time = 0
    
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
    
    status = f"{device_status}\n{ffmpeg_status}\n{model_status}"
    return status

# Main Gradio interface
def create_interface():
    with gr.Blocks(title="Bark Text-to-Speech Generator") as demo:
        gr.Markdown("# 🔊 Bark Text-to-Speech Generator")
        
        # Initialize app
        status = initialize()
        
        with gr.Row():
            with gr.Column(scale=3):
                # Input section
                with gr.Group():
                    gr.Markdown("## Text Input")
                    
                    use_streaming = gr.Checkbox(
                        label="Enable streaming mode", 
                        value=True,
                        info="Generate audio as you type"
                    )
                    
                    reset_btn = gr.Button("Reset Streaming Session")
                    
                    text_input = gr.Textbox(
                        label="Enter text to convert to speech",
                        placeholder="Type your text here...",
                        lines=8
                    )
                    
                    generate_btn = gr.Button("Generate Audio", variant="primary")
                    
                    streaming_info = gr.Markdown(
                        "Streaming mode is enabled. Type a few sentences to automatically generate audio."
                    )
                
                # Output section
                with gr.Group():
                    gr.Markdown("## Audio Output")
                    
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        autoplay=False
                    )
                    status_output = gr.Markdown()
            
            with gr.Column(scale=1):
                # Voice settings
                with gr.Group():
                    gr.Markdown("## Voice Settings")
                    
                    voice_select = gr.Dropdown(
                        choices=list(Voice_presets.keys()),
                        label="Voice",
                        value="Voice 1"
                    )
                    
                    style_select = gr.Dropdown(
                        choices=list(style_presets.keys()),
                        label="Style",
                        value="Default"
                    )
                    
                    style_info = gr.Markdown()
                
                # Advanced parameters
                with gr.Group():
                    gr.Markdown("## Advanced Parameters")
                    
                    pitch_slider = gr.Slider(
                        minimum=-12, 
                        maximum=12, 
                        value=0, 
                        step=1, 
                        label="Pitch adjustment",
                        info="Adjust pitch in semitones (0 = no change)"
                    )
                    
                    speed_slider = gr.Slider(
                        minimum=0.5, 
                        maximum=1.5, 
                        value=1.0, 
                        step=0.1, 
                        label="Speed",
                        info="Adjust speaking speed (1.0 = normal speed)"
                    )
                    
                    text_temp_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1, 
                        label="Text temperature",
                        info="Higher = more creative/variable, Lower = more consistent"
                    )
                    
                    waveform_temp_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1, 
                        label="Waveform temperature",
                        info="Higher = more creative/variable, Lower = more consistent"
                    )
                    
                    add_pauses = gr.Checkbox(
                        label="Add long pauses", 
                        value=True,
                        info="Add longer pauses after punctuation for more natural speech"
                    )
        
        # Update streaming info visibility based on streaming toggle
        def update_streaming_info(use_streaming):
            return gr.update(visible=use_streaming)
            
        use_streaming.change(
            fn=update_streaming_info,
            inputs=[use_streaming],
            outputs=[streaming_info]
        )
        
        # Initialize style info
        style_select.change(
            fn=style_change,
            inputs=[style_select],
            outputs=[text_temp_slider, waveform_temp_slider, style_info]
        )
        
        # Handle reset streaming button
        reset_btn.click(
            fn=reset_streaming_session,
            inputs=[],
            outputs=[status_output]
        )
        
        # Handle text changes for streaming
        text_input.change(
            fn=handle_text_change,
            inputs=[
                text_input, 
                voice_select,
                style_select,
                use_streaming,
                pitch_slider,
                speed_slider,
                text_temp_slider,
                waveform_temp_slider,
                add_pauses
            ],
            outputs=[audio_output, status_output]
        )
        
        # Handle generate button
        generate_btn.click(
            fn=generate_audio_func,
            inputs=[
                text_input, 
                voice_select,
                style_select,
                use_streaming,
                pitch_slider,
                speed_slider,
                text_temp_slider,
                waveform_temp_slider,
                add_pauses
            ],
            outputs=[audio_output, status_output]
        )
        
        # Set initial style info
        demo.load(
            fn=lambda: style_change("Default"),
            inputs=[],
            outputs=[text_temp_slider, waveform_temp_slider, style_info]
        )
        
        # Show initial status
        demo.load(
            fn=lambda: status,
            inputs=[],
            outputs=[status_output]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(share=True) 