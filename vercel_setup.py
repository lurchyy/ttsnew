import os
import logging
import sys

# Configure logging for Vercel environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Helper function to set Vercel-specific environment variables
def configure_vercel_environment():
    # Set environment variables for Vercel
    os.environ['PYTHONPATH'] = os.getcwd()
    
    # AWS Lambda (Vercel uses AWS Lambda) specific settings
    os.environ['TORCH_USE_RTLD_GLOBAL'] = '1'  # Helps with some shared library issues
    
    # Reduce memory usage by limiting the number of threads
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'
    
    # Cache models in /tmp, which is the only writable location in Vercel
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    os.environ['HF_HOME'] = '/tmp/huggingface'
    os.environ['BARK_CACHE_DIR'] = '/tmp/bark_cache'
    
    # Reduce logging from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger.info("Vercel environment configured")

# Run configuration
configure_vercel_environment() 