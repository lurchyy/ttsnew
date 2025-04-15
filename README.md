---
title: tts
app_file: gradio_app.py
sdk: gradio
sdk_version: 5.25.1
---
# Bark Text-to-Speech Generator

An interactive text-to-speech application powered by Suno's Bark AI model and Streamlit.

## Features

- Convert text to natural-sounding speech using state-of-the-art AI
- Multiple voice options with different characteristics
- Streaming mode - hear audio as you type
- Voice style presets (Default, Natural, Expressive)
- Advanced controls:
  - Pitch adjustment
  - Speed adjustment
  - Temperature controls for variability
  - Pause controls for natural speech
- One-click audio download

## Requirements

- Python 3.8+
- Packages listed in `requirements.txt`
- FFmpeg (automatically installed if missing)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-folder>
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run streamlit_app.py
```

## Deployment to Streamlit Cloud

This application is ready to deploy on Streamlit Cloud:

1. Push this repository to GitHub
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, connecting to your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Deploy!

## Usage

1. Enter text in the text area on the left side
2. Choose a voice and style from the options on the right
3. Adjust advanced parameters if desired
4. Click "Generate Audio" or use streaming mode to generate as you type
5. Listen to the generated audio and download if desired

## Performance Notes

- **GPU support**: The application will use GPU acceleration if available, significantly improving generation speed
- **CPU mode**: The app will work on CPU but will be much slower - expect several seconds of processing time per text segment
- **Memory usage**: Bark models require significant memory, especially when using the streaming functionality

## Troubleshooting

- **FFmpeg errors**: The app attempts to install FFmpeg if missing. If you encounter audio processing errors, try installing FFmpeg manually on your system.
- **CUDA errors**: If you have a GPU but encounter CUDA errors, ensure you have the correct CUDA toolkit installed for your version of PyTorch.
- **Memory errors**: If you encounter out-of-memory errors, try reducing the size of text being processed at once, or run the app on a machine with more memory.

## Credits

- Built with [Streamlit](https://streamlit.io/)
- Text-to-speech powered by [Suno's Bark](https://github.com/suno-ai/bark)
- Audio processing using [pydub](https://github.com/jiaaro/pydub)

## License

[MIT License](LICENSE) 