# Bark Text-to-Speech Generator

A web application for generating natural-sounding speech from text using the Bark text-to-speech model. This application provides a user-friendly interface for converting text to speech with various voice options and customization settings.

## Features

- Text-to-speech generation with 10 different voice options
- Voice style presets (Default, Natural, Expressive)
- Pitch and speed adjustment
- Advanced parameter tuning
- Audio download capability
- CPU/GPU compatibility with automatic detection

## Requirements

- Python 3.8+
- Flask
- PyTorch
- Bark TTS model
- FFmpeg

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

   For production deployment:
   ```
   python wsgi.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8080`
2. Enter the text you want to convert to speech
3. Select a voice and adjust parameters as desired
4. Click "Generate Audio" to create your speech
5. Play the audio in the browser or download it

## API Endpoints

The application provides the following REST API endpoints:

- `GET /api/voices` - Get a list of available voices
- `GET /api/styles` - Get a list of available voice styles
- `POST /api/generate` - Generate audio from text
- `POST /api/download` - Download generated audio
- `GET /health` - Health check endpoint

### Example API Request

```javascript
fetch('/api/generate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: "Hello, world!",
        voice: "Voice 1",
        pitch: 0,
        speed: 1.0,
        style: "natural",
        text_temp: 0.7,
        waveform_temp: 0.7,
        add_pauses: true
    })
})
.then(response => response.json())
.then(data => {
    console.log(data);
});
```

## Performance Notes

- GPU mode is significantly faster than CPU mode
- When running in CPU mode, processing longer texts will take more time
- The first generation after loading the app may take longer as models are loaded

## Credits

This application uses the [Bark](https://github.com/suno-ai/bark) text-to-speech model by Suno AI.

## License

[MIT License](LICENSE) 