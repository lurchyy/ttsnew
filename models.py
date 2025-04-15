import torch
import numpy as np
import logging
from bark import preload_models
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
models_loaded = False

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

# Optimize memory usage for CPU mode
def optimize_for_cpu():
    if device == "cpu":
        # Apply memory optimizations when running on CPU
        torch.set_num_threads(max(4, os.cpu_count() or 4))  # Set reasonable thread count
        logger.info(f"CPU mode: Using {torch.get_num_threads()} CPU threads") 