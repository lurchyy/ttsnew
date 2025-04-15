import torch
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import display
import IPython.display as ipd
import ipywidgets as widgets
from pydub import AudioSegment
import os
import threading
import time
import glob

# 🛠 Fix for Torch compatibility
torch.serialization.add_safe_globals({
    "numpy.core.multiarray.scalar": np.ndarray
})

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# 📦 Load Bark models
preload_models()

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

# 🔘 Voice selection dropdown
Voice_dropdown = widgets.Dropdown(
    options=Voice_presets,
    value="v2/en_speaker_1",
    description='Choose Voice:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='350px'),
)

# 📝 Text input
text_input = widgets.Textarea(
    value="Hello! I'm speaking with a human-like voice and a selected Voice.",
    placeholder='Type something to say...',
    description='Text:',
    layout=widgets.Layout(width='100%', height='100px'),
    style={'description_width': 'initial'}
)

# 🎚️ Pitch and Speed sliders
pitch_slider = widgets.IntSlider(
    value=0, min=-10, max=10, step=1, description='Pitch (semitones):'
)
speed_slider = widgets.FloatSlider(
    value=1.0, min=0.5, max=1.5, step=0.1, description='Speed:'
)

# 🎛️ Toggle for streaming mode
streaming_toggle = widgets.ToggleButton(
    value=True,
    description='Streaming Mode: ON',
    disabled=False,
    button_style='success',
    tooltip='Toggle streaming mode on/off',
    icon='microphone'
)

# ▶️ Generate button
generate_button = widgets.Button(description="Generate Audio 🎤")
download_button = widgets.Button(description="Download Audio 💾", button_style='success', disabled=True)

# 🎧 Output area
output = widgets.Output()

# 🔄 Global variables for streaming
is_processing = False
last_generation_time = 0
previous_text = ""
last_text = ""
last_voice = Voice_dropdown.value
debounce_time = 1  # seconds
current_audio = None  # To store the latest audio for download

# Check if running in IPython/Jupyter environment
try:
    get_ipython
    in_jupyter = True
except NameError:
    in_jupyter = False

# Create a function to force audio playback - removing this function as it requires JavaScript
def force_autoplay():
    # Function disabled as it requires JavaScript
    return False

# Add download functionality
def on_download_clicked(b):
    global current_audio
    if current_audio is not None:
        with output:
            output.clear_output()
            print("💾 Preparing audio for download...")
        
        # Create a downloadable audio file
        filename = f"bark_voice_{list(Voice_presets.keys())[list(Voice_presets.values()).index(Voice_dropdown.value)]}.wav"
        
        # Convert current_audio to downloadable format if needed
        if isinstance(current_audio, np.ndarray):
            # It's a raw numpy array - create and display audio widget with data
            with output:
                print(f"✅ Download ready: {filename}")
                display(ipd.Audio(data=current_audio, rate=SAMPLE_RATE, autoplay=False))
        else:
            # It's already processed audio (likely a bytes object)
            with output:
                print(f"✅ Download ready: {filename}")
                # Create audio widget directly from data, not from filename
                display(ipd.Audio(data=current_audio, rate=SAMPLE_RATE, autoplay=False))

# Add advanced generation parameters for more natural voices
generation_temp = widgets.FloatSlider(
    value=0.7, 
    min=0.1, 
    max=1.0, 
    step=0.05,
    description='Temperature:',
    tooltip='Lower = more deterministic, Higher = more varied/creative',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='70%')
)

generation_waveform_temp = widgets.FloatSlider(
    value=0.7, 
    min=0.1, 
    max=1.0, 
    step=0.05,
    description='Waveform Temp:',
    tooltip='Controls variability in the audio waveform',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='70%')
)

# Add pause controls
add_long_pauses = widgets.Checkbox(
    value=True,
    description='Natural Pauses',
    tooltip='Add longer pauses at punctuation (commas, periods) for more natural speech',
    style={'description_width': 'initial'}
)

# Add voice style presets
voice_style_dropdown = widgets.Dropdown(
    options=[
        ('Default', 'default'),
        ('Natural', 'natural'),
        ('Expressive', 'expressive'),
        ('Calm', 'calm'),
        ('Precise', 'precise')
    ],
    value='default',
    description='Voice Style:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='25%')
)

# Button to apply voice style presets
apply_style_button = widgets.Button(
    description="Apply Style",
    button_style='info',
    tooltip='Apply selected voice style preset'
)

# Apply voice style presets
def apply_voice_style(b):
    global is_processing
    style = voice_style_dropdown.value
    current_text = text_input.value
    
    with output:
        output.clear_output()
        print(f"🎭 Applying voice style: {style}")
    
    # If there's an ongoing generation, cancel it
    if is_processing:
        # Signal to the generation function that it should stop
        is_processing = False
        print("⏹️ Stopping current generation...")
        # Give the generation function time to cancel
        time.sleep(1)
    
    # Apply the selected style parameters
    if style == 'default':
        generation_temp.value = 0.7
        generation_waveform_temp.value = 0.7
        add_long_pauses.value = True
    elif style == 'natural':
        generation_temp.value = 0.6
        generation_waveform_temp.value = 0.5
        add_long_pauses.value = True
    elif style == 'expressive':
        generation_temp.value = 0.9
        generation_waveform_temp.value = 0.8
        add_long_pauses.value = True
    # elif style == 'calm':
    #     generation_temp.value = 0.4
    #     generation_waveform_temp.value = 0.4
    #     add_long_pauses.value = True
    # elif style == 'precise':
    #     generation_temp.value = 0.3
    #     generation_waveform_temp.value = 0.2
    #     add_long_pauses.value = False
    
    # Generate audio with the new style settings if there's text
    if current_text.strip():
        # Small delay to allow UI updates before generation
        time.sleep(0.5)
        generate_audio_from_text(current_text)

apply_style_button.on_click(apply_voice_style)

# Modify text to add more natural pauses if needed
def process_text_for_speech(text):
    if not add_long_pauses.value:
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
def generate_audio_from_text(text):
    global is_processing, last_generation_time, current_audio
    
    # Make sure we don't have an ongoing process
    if is_processing:
        return
    
    # Reset any ongoing processes (in case another function is trying to use this)
    is_processing = True
    last_generation_time = time.time()
    download_button.disabled = True
    
    # Continue with generation as before
    with output:
        output.clear_output()
        selected_label = [k for k, v in Voice_presets.items() if v == Voice_dropdown.value][0]
        print(f"🗣️ Using Voice: {selected_label}")
        print(f"🎭 Style: {voice_style_dropdown.value}")
        print(f"⏳ Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"🎛️ Using temperature: {generation_temp.value}, waveform temp: {generation_waveform_temp.value}")
        if add_long_pauses.value:
            print("👂 Adding natural pauses at punctuation")

        try:
            # Process text for more natural speech with pauses
            processed_text = process_text_for_speech(text)
            
            # Check if we've been interrupted
            if not is_processing:
                print("❌ Generation canceled")
                return
                
            # 🔊 Generate raw Bark audio with adjusted parameters
            audio_array = generate_audio(
                processed_text, 
                history_prompt=Voice_dropdown.value,
                text_temp=generation_temp.value,
                waveform_temp=generation_waveform_temp.value
            )
            
            # Check if we've been interrupted
            if not is_processing:
                print("❌ Generation canceled")
                return
                
            current_audio = audio_array  # Store raw audio for possible download
            
            # Apply pitch and speed adjustments directly to the numpy array
            if pitch_slider.value != 0 or speed_slider.value != 1.0:
                print("✨ Applying voice adjustments...")
                
                # Convert numpy array to AudioSegment for processing
                import io
                import tempfile
                
                # Create a temporary in-memory WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                    # Write audio data to temp file
                    write_wav(temp_file.name, rate=SAMPLE_RATE, data=audio_array)
                    
                    # Apply adjustments
                    sound = AudioSegment.from_wav(temp_file.name)
                    
                    # ⏩ Adjust speed
                    if speed_slider.value != 1.0:
                        new_frame_rate = int(sound.frame_rate * speed_slider.value)
                        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
                        sound = sound.set_frame_rate(SAMPLE_RATE)

                    # 🎵 Adjust pitch
                    if pitch_slider.value != 0:
                        pitch_factor = 2 ** (pitch_slider.value / 12)
                        new_rate = int(sound.frame_rate * pitch_factor)
                        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_rate})
                        sound = sound.set_frame_rate(SAMPLE_RATE)
                    
                    # Check if we've been interrupted during processing
                    if not is_processing:
                        print("❌ Processing canceled")
                        return
                    
                    # Export to in-memory file
                    buffer = io.BytesIO()
                    sound.export(buffer, format="wav")
                    buffer.seek(0)
                    
                    # Update current_audio with adjusted version
                    current_audio = buffer.read()
                    buffer.seek(0)
                    
                    # Display audio widget - always with autoplay=False 
                    display(ipd.Audio(buffer.read(), rate=SAMPLE_RATE, autoplay=False))
            else:
                # Play the raw audio directly without adjustments
                display(ipd.Audio(audio_array, rate=SAMPLE_RATE, autoplay=False))
            
            # Check if we've been interrupted before attempting autoplay
            if not is_processing:
                print("❌ Processing canceled before playback")
                return
            
            print("✅ Audio ready to play")
            download_button.disabled = False
            
        except Exception as e:
            print(f"❌ Error generating audio: {str(e)}")
    
    is_processing = False

# 🎯 Bind button click
def on_generate_clicked(b):
    generate_audio_from_text(text_input.value)

# Update the streaming toggle description when clicked
def on_streaming_toggle_change(change):
    if change['new']:
        streaming_toggle.description = 'Streaming Mode: ON'
        streaming_toggle.button_style = 'success'
        with output:
            print("🔄 Streaming mode ON - Audio will be generated automatically")
    else:
        streaming_toggle.description = 'Streaming Mode: OFF'
        streaming_toggle.button_style = 'danger'
        with output:
            print("⏸️ Streaming mode OFF - Manual generation required")

# Handle voice change - generate audio when voice changes
def on_voice_change(change):
    if streaming_toggle.value and not is_processing:
        with output:
            print(f"🎤 Voice changed to: {change['new']}")
        generate_audio_from_text(text_input.value)

# ⌨️ Streaming logic with proper debouncing for Jupyter
def streaming_check():
    """Check for text changes periodically and trigger audio generation"""
    global previous_text, last_generation_time, last_text, last_voice
    
    # Only run if streaming is enabled and not currently processing
    if streaming_toggle.value and not is_processing:
        current_text = text_input.value
        current_voice = Voice_dropdown.value
        now = time.time()
        
        # Check for voice change
        voice_changed = current_voice != last_voice
        if voice_changed:
            last_voice = current_voice
            with output:
                print(f"🎤 Voice changed to: {current_voice}")
            generate_audio_from_text(current_text)
            return
        
        # Only process if text has changed and it's significantly different
        if (current_text != last_text and 
            len(current_text) > 5 and  # Minimum content length
            (now - last_generation_time) > debounce_time):
            
            with output:
                print(f"🔄 Streaming detected text change: '{current_text[:20]}...'")
            
            last_text = current_text
            last_generation_time = now
            
            # Generate audio for the new text
            generate_audio_from_text(current_text)
    
    # Schedule the next check (regular polling approach that works in Jupyter)
    threading.Timer(1.0, streaming_check).start()

# Initialize the app and register callbacks
generate_button.on_click(on_generate_clicked)
streaming_toggle.observe(on_streaming_toggle_change, names='value')
Voice_dropdown.observe(on_voice_change, names='value')
download_button.on_click(on_download_clicked)

# Update the UI layout to include the new controls
display(
    widgets.HTML("<h2>🔊 Bark Text-to-Speech Generator (English)</h2>"),
    Voice_dropdown,
    text_input,
    widgets.HTML("<h3>Voice Settings</h3>"),
    widgets.HBox([pitch_slider, speed_slider], layout=widgets.Layout(width='100%')),
    widgets.HTML("<h3>Voice Style Presets</h3>"),
    widgets.HBox([voice_style_dropdown, apply_style_button], layout=widgets.Layout(width='100%')),
    widgets.HTML("<h3>Advanced Parameters</h3>"),
    widgets.HBox([
        widgets.VBox([generation_temp, generation_waveform_temp], layout=widgets.Layout(width='80%')),
        widgets.VBox([add_long_pauses], layout=widgets.Layout(width='20%'))
    ], layout=widgets.Layout(width='100%')),
    widgets.HTML("<h3>Controls</h3>"),
    widgets.HBox([streaming_toggle, generate_button, download_button], layout=widgets.Layout(width='100%')),
    widgets.HTML("<h3>Output</h3>"),
    output
)

# Start the streaming checker
streaming_check()

# Add cleanup for any existing files
def cleanup_audio_files():
    try:
        bark_files = glob.glob("bark_output_*.wav")
        if bark_files:
            print(f"🧹 Cleaning up {len(bark_files)} old audio files...")
            for file in bark_files:
                try:
                    os.remove(file)
                except Exception:
                    pass
    except Exception:
        pass  # Ignore any errors during cleanup

# Run cleanup at startup
cleanup_audio_files()
