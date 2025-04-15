// Global variables
let currentAudio = null;
let currentStreamingSession = null;
let streamingDebounceTimeout = null;
let isStreamingMode = true;
let streamingInProgress = false;
const debounceDelay = 800; // ms to wait after typing before generating audio
const minStreamingTextLength = 8; // Minimum text length before we start streaming
let audioPlayer = null;
let isGenerating = false;
let selectedVoice = '';
let streamingEnabled = false;
let streamingTimeout = null;
let lastStreamingText = '';

// Constants for streaming 
const STREAMING_DEBOUNCE_MS = 2000;  // Wait time after typing before generating
const MIN_NEW_TEXT_LENGTH = 5;       // Minimum new text length to trigger streaming

// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const textInput = document.getElementById('text-input');
    const voiceSelect = document.getElementById('voice-select');
    const styleSelect = document.getElementById('style-select');
    const pitchInput = document.getElementById('pitch');
    const speedInput = document.getElementById('speed');
    const textTempInput = document.getElementById('text-temp');
    const waveformTempInput = document.getElementById('waveform-temp');
    const addPausesInput = document.getElementById('add-pauses');
    const streamingToggle = document.getElementById('streaming-toggle');
    const streamingIndicator = document.getElementById('streaming-indicator');
    const generateButton = document.getElementById('generate-button');
    const downloadButton = document.getElementById('download-button');
    const resetStreamingButton = document.getElementById('reset-streaming-button');
    const statusDiv = document.getElementById('status');
    const audioPlayer = document.getElementById('audio-player');
    
    // Variables for streaming
    let lastStreamingText = '';
    let streamingSessionId = null;
    let streamingTimeout = null;
    const STREAMING_TYPING_DELAY = 1000; // ms to wait after typing stops before sending request
    
    // Load voices
    loadVoices();
    
    // Load styles
    loadStyles();
    
    // Set up event listeners
    generateButton.addEventListener('click', function() {
        if (streamingToggle.checked) {
            // If streaming is enabled but user clicks generate button,
            // generate audio for current text without waiting for typing pause
            clearTimeout(streamingTimeout);
            generateStreamingAudio(true);
        } else {
            generateAudio();
        }
    });
    
    downloadButton.addEventListener('click', function() {
        downloadAudio();
    });
    
    resetStreamingButton.addEventListener('click', function() {
        resetStreamingSession();
    });
    
    styleSelect.addEventListener('change', function() {
        applyStyle(this.value);
    });
    
    // Set up streaming mode
    textInput.addEventListener('input', function() {
        if (streamingToggle.checked) {
            // Clear existing timeout
            clearTimeout(streamingTimeout);
            
            // Set streaming indicator
            streamingIndicator.style.display = 'inline-block';
            
            // Set a timeout to wait for typing to pause
            streamingTimeout = setTimeout(function() {
                generateStreamingAudio();
            }, STREAMING_TYPING_DELAY);
        }
    });
    
    streamingToggle.addEventListener('change', function() {
        if (this.checked) {
            streamingIndicator.style.display = 'inline-block';
            resetStreamingButton.style.display = 'inline-block';
            // Initialize streaming session if not already done
            if (!streamingSessionId) {
                resetStreamingSession();
            }
        } else {
            streamingIndicator.style.display = 'none';
            resetStreamingButton.style.display = 'none';
        }
    });
    
    // Initialize UI
    downloadButton.disabled = true;
    resetStreamingButton.style.display = 'none';
    streamingIndicator.style.display = 'none';
    
    // Apply default style
    applyStyle('default');
});

function loadVoices() {
    fetch('/api/voices')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const voiceSelect = document.getElementById('voice-select');
                voiceSelect.innerHTML = '';
                
                data.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice.id;
                    option.textContent = voice.name;
                    voiceSelect.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('Error loading voices:', error);
            showStatus('Error loading voices. Please try refreshing the page.', 'error');
        });
}

function loadStyles() {
    fetch('/api/styles')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const styleSelect = document.getElementById('style-select');
                styleSelect.innerHTML = '';
                
                data.styles.forEach(style => {
                    const option = document.createElement('option');
                    option.value = style.id;
                    option.textContent = style.name;
                    styleSelect.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('Error loading styles:', error);
            showStatus('Error loading styles. Please try refreshing the page.', 'error');
        });
}

function applyStyle(style) {
    fetch(`/api/apply_style?style=${style}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI with style parameters
                document.getElementById('text-temp').value = data.parameters.text_temp;
                document.getElementById('waveform-temp').value = data.parameters.waveform_temp;
                document.getElementById('pitch').value = data.parameters.pitch;
                document.getElementById('speed').value = data.parameters.speed;
                
                // Show style info
                fetch(`/api/style_info?style=${style}`)
                    .then(response => response.json())
                    .then(infoData => {
                        if (infoData.success) {
                            showStatus(`Style applied: ${infoData.info}`, 'info');
                        }
                    });
            } else {
                showStatus('Failed to apply style.', 'error');
            }
        })
        .catch(error => {
            console.error('Error applying style:', error);
            showStatus('Error applying style.', 'error');
        });
}

function resetStreamingSession() {
    // Clear any existing timeout
    clearTimeout(streamingTimeout);
    
    // Reset last streaming text
    lastStreamingText = '';
    
    // Generate a new session ID
    streamingSessionId = null;
    
    // Update streaming indicator
    const streamingIndicator = document.getElementById('streaming-indicator');
    streamingIndicator.style.display = document.getElementById('streaming-toggle').checked ? 'inline-block' : 'none';
    
    // Reset the audio player
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = '';
    audioPlayer.style.display = 'none';
    
    // Reset download button
    document.getElementById('download-button').disabled = true;
    
    // Reset status
    showStatus('Streaming session reset. Start typing to generate audio.', 'info');
    
    // Call API to reset session
    fetch('/api/reset_streaming', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: streamingSessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            streamingSessionId = data.session_id;
            console.log('Streaming session reset with ID:', streamingSessionId);
        }
    })
    .catch(error => {
        console.error('Error resetting streaming session:', error);
    });
}

function generateStreamingAudio(forceGenerate = false) {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    // Only proceed if streaming is enabled
    if (!document.getElementById('streaming-toggle').checked) {
        return;
    }
    
    // Check if there's any text to process
    if (!text) {
        return;
    }
    
    // Check if there's new text to process
    if (!forceGenerate && text === lastStreamingText) {
        // No new text, so don't generate
        return;
    }
    
    // Update the streaming indicator
    const streamingIndicator = document.getElementById('streaming-indicator');
    streamingIndicator.classList.add('active');
    
    // Show status
    showStatus('Generating audio...', 'info');
    
    // Get user parameters
    const voicePreset = document.getElementById('voice-select').value;
    const style = document.getElementById('style-select').value;
    const pitch = parseInt(document.getElementById('pitch').value, 10);
    const speed = parseFloat(document.getElementById('speed').value);
    const textTemp = parseFloat(document.getElementById('text-temp').value);
    const waveformTemp = parseFloat(document.getElementById('waveform-temp').value);
    const addPauses = document.getElementById('add-pauses').checked;
    
    // Make API request
    fetch('/api/generate_streaming', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: streamingSessionId,
            text: text,
            voice_preset: voicePreset,
            style: style,
            pitch: pitch,
            speed: speed,
            text_temp: textTemp,
            waveform_temp: waveformTemp,
            add_pauses: addPauses
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update session ID if it's been created or changed
            if (data.session_id) {
                streamingSessionId = data.session_id;
            }
            
            // Update last streaming text
            lastStreamingText = text;
            
            // If there's audio data, update the player
            if (data.audio_data) {
                updateAudioPlayer(data.audio_data, data.format);
                showStatus('Audio updated.', 'success');
            } else {
                showStatus('No new audio to generate.', 'info');
            }
        } else {
            showStatus('Failed to generate audio: ' + (data.error || 'Unknown error'), 'error');
        }
        
        // Update the streaming indicator
        streamingIndicator.classList.remove('active');
    })
    .catch(error => {
        console.error('Error generating streaming audio:', error);
        showStatus('Error generating audio. Please try again.', 'error');
        streamingIndicator.classList.remove('active');
    });
}

function generateAudio() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    if (!text) {
        showStatus('Please enter some text to generate audio.', 'warning');
        return;
    }
    
    // Show status
    showStatus('Generating audio...', 'info');
    
    // Disable generate button during generation
    const generateButton = document.getElementById('generate-button');
    generateButton.disabled = true;
    generateButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    
    // Get user parameters
    const voicePreset = document.getElementById('voice-select').value;
    const style = document.getElementById('style-select').value;
    const pitch = parseInt(document.getElementById('pitch').value, 10);
    const speed = parseFloat(document.getElementById('speed').value);
    const textTemp = parseFloat(document.getElementById('text-temp').value);
    const waveformTemp = parseFloat(document.getElementById('waveform-temp').value);
    const addPauses = document.getElementById('add-pauses').checked;
    
    // Make API request
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            voice_preset: voicePreset,
            style: style,
            pitch: pitch,
            speed: speed,
            text_temp: textTemp,
            waveform_temp: waveformTemp,
            add_pauses: addPauses
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateAudioPlayer(data.audio_data, data.format);
            showStatus('Audio generated successfully.', 'success');
        } else {
            showStatus('Failed to generate audio: ' + (data.error || 'Unknown error'), 'error');
        }
        
        // Re-enable generate button
        generateButton.disabled = false;
        generateButton.innerHTML = 'Generate Audio';
    })
    .catch(error => {
        console.error('Error generating audio:', error);
        showStatus('Error generating audio. Please try again.', 'error');
        
        // Re-enable generate button
        generateButton.disabled = false;
        generateButton.innerHTML = 'Generate Audio';
    });
}

function updateAudioPlayer(audioData, format) {
    const audioPlayer = document.getElementById('audio-player');
    const downloadButton = document.getElementById('download-button');
    
    // Convert base64 to blob
    const audioBlob = base64ToBlob(audioData, 'audio/' + format);
    
    // Create object URL
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Update audio player
    audioPlayer.src = audioUrl;
    audioPlayer.style.display = 'block';
    
    // Enable download button
    downloadButton.disabled = false;
    
    // Play audio automatically
    audioPlayer.play().catch(error => {
        console.error('Error playing audio:', error);
    });
}

function downloadAudio() {
    const audioPlayer = document.getElementById('audio-player');
    
    if (!audioPlayer.src) {
        showStatus('No audio available to download.', 'warning');
        return;
    }
    
    // Create an anchor element
    const a = document.createElement('a');
    a.href = audioPlayer.src;
    a.download = 'generated_audio.wav';
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteArrays = [];
    
    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
    }
    
    return new Blob(byteArrays, { type: mimeType });
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    
    // Set message and class
    statusDiv.textContent = message;
    statusDiv.className = 'alert';
    
    // Add appropriate class based on message type
    switch (type) {
        case 'success':
            statusDiv.classList.add('alert-success');
            break;
        case 'info':
            statusDiv.classList.add('alert-info');
            break;
        case 'warning':
            statusDiv.classList.add('alert-warning');
            break;
        case 'error':
            statusDiv.classList.add('alert-danger');
            break;
        default:
            statusDiv.classList.add('alert-info');
    }
    
    // Show status
    statusDiv.style.display = 'block';
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success' || type === 'info') {
        setTimeout(function() {
            statusDiv.style.display = 'none';
        }, 5000);
    }
} 