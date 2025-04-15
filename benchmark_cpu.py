import torch
import numpy as np
import time
import gc
import os
import tempfile
import argparse
import io
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav
from bark import generate_audio, SAMPLE_RATE, preload_models

# Set device to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"
torch.set_default_device("cpu")

# Configure CPU threads based on system resources
cpu_count = os.cpu_count() or 4
torch.set_num_threads(max(4, cpu_count))
print(f"Using {torch.get_num_threads()} CPU threads")

# Apply torch compatibility fixes
def setup_torch_compatibility():
    try:
        torch.serialization.add_safe_globals({
            "numpy.core.multiarray.scalar": np.ndarray
        })
    except AttributeError:
        print("Warning: Could not add safe globals to torch.serialization")

    # Patch torch.load to avoid weights_only warning
    try:
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
    except Exception as e:
        print(f"Warning: Could not patch torch.load: {str(e)}")

# Memory cleanup function for CPU
def cleanup_memory():
    gc.collect()
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
    gc.collect()

# Selected Voice Presets
Voice_presets = {
    "voice_1": "v2/en_speaker_1",
    "voice_2": "v2/en_speaker_2",
    "voice_3": "v2/en_speaker_3",
    "voice_9": "v2/en_speaker_9",
    "british": "v2/en_speaker_0",
}

def process_text_for_speech(text, add_long_pauses=True):
    if not add_long_pauses:
        return text
    
    # Add longer pauses at punctuation
    text = text.replace('. ', '.  ')  # Double space after periods
    text = text.replace('! ', '!  ')  # Double space after exclamation marks
    text = text.replace('? ', '?  ')  # Double space after question marks
    text = text.replace(', ', ',  ')  # Double space after commas
    text = text.replace('; ', ';  ')  # Double space after semicolons
    
    return text

def generate_audio_segment(text, voice_preset, pitch=0, speed=1.0, text_temp=0.7, waveform_temp=0.7, add_long_pauses=True):
    if not text.strip():
        return None
        
    # Process text for more natural speech with pauses
    processed_text = process_text_for_speech(text, add_long_pauses)
    
    # Track generation time
    start_time = time.time()
    
    # Generate raw Bark audio with adjusted parameters
    with torch.device(device):
        audio_array = generate_audio(
            processed_text, 
            history_prompt=voice_preset,
            text_temp=text_temp,
            waveform_temp=waveform_temp
        )
    
    generation_time = time.time() - start_time
    
    # Time the pitch and speed adjustments
    adjust_start_time = time.time()
    
    # Apply pitch and speed adjustments if needed
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
                
            # Export back to audio array
            buffer = io.BytesIO()
            sound.export(buffer, format="wav")
            buffer.seek(0)
            audio_data = buffer.read()
    else:
        # Just convert to bytes
        with io.BytesIO() as buffer:
            write_wav(buffer, rate=SAMPLE_RATE, data=audio_array)
            buffer.seek(0)
            audio_data = buffer.read()
            
    adjust_time = time.time() - adjust_start_time
    
    # Clean up memory
    cleanup_memory()
    
    return audio_data, generation_time, adjust_time

def run_benchmark(text_lengths=[50, 100, 200, 500], voice_presets=None, iterations=1):
    """Run CPU benchmark with various text lengths and voice presets"""
    if voice_presets is None:
        # Default to using just one voice for benchmark
        voice_presets = {"voice_1": Voice_presets["voice_1"]}
    
    print("\n============================================")
    print("BARK TEXT-TO-SPEECH CPU BENCHMARK")
    print("============================================")
    print(f"CPU: {torch.get_num_threads()} threads")
    print(f"Pytorch version: {torch.__version__}")
    print("============================================\n")
    
    # Load models (time this as well)
    print("Loading models (this may take a while)...")
    load_start = time.time()
    preload_models()
    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Prepare test texts
    texts = {}
    for length in text_lengths:
        if length <= 100:
            # Generate a repeating pattern to reach desired length
            texts[length] = f"This is a test sentence for benchmark purposes. " * (length // 10 + 1)
            texts[length] = texts[length][:length]
        else:
            # For longer texts, use lorem ipsum to avoid repetition
            texts[length] = f"""
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
            eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt 
            in culpa qui officia deserunt mollit anim id est laborum.
            """ * (length // 100 + 1)
            texts[length] = texts[length].replace('\n', ' ').replace('            ', '').strip()[:length]
    
    # Run the benchmark
    results = []
    
    print("\nRunning benchmark tests...\n")
    
    # Run each test case
    for voice_name, voice_preset in voice_presets.items():
        for text_length, text in texts.items():
            # Average over multiple iterations
            total_gen_time = 0
            total_adjust_time = 0
            success_count = 0
            
            for i in range(iterations):
                print(f"Testing voice: {voice_name}, length: {text_length} chars, iteration {i+1}/{iterations}")
                try:
                    _, gen_time, adj_time = generate_audio_segment(
                        text, 
                        voice_preset=voice_preset,
                        pitch=0,
                        speed=1.0,
                        text_temp=0.7,
                        waveform_temp=0.7
                    )
                    total_gen_time += gen_time
                    total_adjust_time += adj_time
                    success_count += 1
                    print(f"  Generation time: {gen_time:.2f}s, Adjustment time: {adj_time:.2f}s")
                except Exception as e:
                    print(f"Error during generation: {str(e)}")
            
            # Record results only if at least one iteration succeeded
            if success_count > 0:
                avg_gen_time = total_gen_time / success_count
                avg_adjust_time = total_adjust_time / success_count
                results.append({
                    "voice": voice_name,
                    "length": text_length,
                    "chars_per_second": text_length / avg_gen_time,
                    "generation_time": avg_gen_time,
                    "adjustment_time": avg_adjust_time,
                    "total_time": avg_gen_time + avg_adjust_time
                })
    
    # Print results summary
    print("\n============================================")
    print("BENCHMARK RESULTS")
    print("============================================")
    print(f"{'Voice':<10} {'Length':<8} {'Gen. Time':<12} {'Adj. Time':<12} {'Total':<10} {'Chars/sec':<10}")
    print("--------------------------------------------")
    
    for r in results:
        print(f"{r['voice']:<10} {r['length']:<8} {r['generation_time']:.2f}s{'':<6} {r['adjustment_time']:.2f}s{'':<6} {r['total_time']:.2f}s{'':<4} {r['chars_per_second']:.2f}")
    
    print("============================================")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Bark TTS on CPU')
    parser.add_argument('--lengths', type=int, nargs='+', default=[50, 100, 200], 
                       help='Text lengths to test (default: 50 100 200)')
    parser.add_argument('--voices', type=str, nargs='+', default=['voice_1'],
                       help='Voice presets to test (default: voice_1). Options: voice_1, voice_2, voice_3, voice_9, british')
    parser.add_argument('--iterations', type=int, default=1, 
                       help='Number of iterations per test (default: 1)')
    
    args = parser.parse_args()
    
    # Setup torch compatibility
    setup_torch_compatibility()
    
    # Select voice presets for testing
    selected_voices = {}
    for v in args.voices:
        if v in Voice_presets:
            selected_voices[v] = Voice_presets[v]
        else:
            print(f"Warning: Unknown voice preset '{v}'. Skipping.")
            
    if not selected_voices:
        print("Error: No valid voice presets specified. Using default voice_1.")
        selected_voices = {"voice_1": Voice_presets["voice_1"]}
    
    # Run the benchmark
    run_benchmark(
        text_lengths=args.lengths,
        voice_presets=selected_voices,
        iterations=args.iterations
    ) 