// Global variables
let currentAudio = null;
let currentStreamingSession = null;
let streamingDebounceTimeout = null;
let isStreamingMode = true;
let streamingInProgress = false;
const debounceDelay = 800; // ms to wait after typing before generating audio
const minStreamingTextLength = 8; // Minimum text length before we start streaming

// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const textInput = document.getElementById('textToSpeak');
    const voiceSelect = document.getElementById('voiceSelection');
    const styleSelect = document.getElementById('voiceStyle');
    const pitchSlider = document.getElementById('pitch');
    const speedSlider = document.getElementById('speed');
    const textTempSlider = document.getElementById('textTemp');
    const waveformTempSlider = document.getElementById('waveformTemp');
    const addPausesCheckbox = document.getElementById('addPauses');
    const generateBtn = document.getElementById('generateBtn');
    const resetBtn = document.getElementById('resetBtn');
    const streamingToggle = document.getElementById('streamingToggle');
    const streamingIndicator = document.getElementById('streamingIndicator');
    const progressContainer = document.getElementById('progressContainer');
    const progressText = document.getElementById('progressText');
    const audioContainer = document.getElementById('audioContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const downloadBtn = document.getElementById('downloadBtn');
    const statusMessage = document.getElementById('statusMessage');
    const styleInfo = document.getElementById('styleInfo');
    
    // Value displays
    const pitchValue = document.getElementById('pitchValue');
    const speedValue = document.getElementById('speedValue');
    const textTempValue = document.getElementById('textTempValue');
    const waveformTempValue = document.getElementById('waveformTempValue');
    
    // Update value displays
    pitchSlider.addEventListener('input', () => pitchValue.textContent = pitchSlider.value);
    speedSlider.addEventListener('input', () => speedValue.textContent = speedSlider.value);
    textTempSlider.addEventListener('input', () => textTempValue.textContent = textTempSlider.value);
    waveformTempSlider.addEventListener('input', () => waveformTempValue.textContent = waveformTempSlider.value);
    
    // Toggle streaming mode
    streamingToggle.addEventListener('change', () => {
        isStreamingMode = streamingToggle.checked;
        resetStreamingSession(); // Always reset when toggling
        
        if (isStreamingMode) {
            streamingIndicator.style.display = 'block';
            resetBtn.style.display = 'block';
            generateBtn.textContent = 'Generate Now 🎤';
            
            // Start streaming with current text if long enough
            if (textInput.value.trim().length >= minStreamingTextLength) {
                processStreamingText();
            }
        } else {
            streamingIndicator.style.display = 'none';
            resetBtn.style.display = 'none';
            generateBtn.textContent = 'Generate Audio 🎤';
        }
    });
    
    // Text input streaming
    textInput.addEventListener('input', () => {
        if (isStreamingMode && !streamingInProgress && textInput.value.trim().length >= minStreamingTextLength) {
            // Clear any existing timeout
            if (streamingDebounceTimeout) {
                clearTimeout(streamingDebounceTimeout);
            }
            
            // Set new timeout to debounce rapid typing
            streamingDebounceTimeout = setTimeout(() => {
                processStreamingText();
            }, debounceDelay);
        }
    });
    
    // Voice or style changes should reset streaming
    voiceSelect.addEventListener('change', () => {
        if (isStreamingMode) {
            resetStreamingSession();
            // Regenerate with new voice if text exists
            if (textInput.value.trim().length >= minStreamingTextLength) {
                processStreamingText();
            }
        }
    });
    
    // Load voices
    loadVoices();
    
    // Load styles
    loadStyles();
    
    // Style changes
    styleSelect.addEventListener('change', () => {
        updateStyleInfo();
        applyStyle(styleSelect.value);
        
        // Regenerate with new style if streaming is on
        if (isStreamingMode && textInput.value.trim().length >= minStreamingTextLength) {
            resetStreamingSession();
            processStreamingText();
        }
    });
    
    // Parameter changes
    pitchSlider.addEventListener('change', parameterChangeHandler);
    speedSlider.addEventListener('change', parameterChangeHandler);
    textTempSlider.addEventListener('change', parameterChangeHandler);
    waveformTempSlider.addEventListener('change', parameterChangeHandler);
    addPausesCheckbox.addEventListener('change', parameterChangeHandler);
    
    // Generate audio button
    generateBtn.addEventListener('click', () => {
        if (isStreamingMode) {
            // In streaming mode, generate or regenerate immediately
            resetStreamingSession();
            processStreamingText(true); // force immediate generation
        } else {
            // In normal mode, use regular generation
            generateAudio();
        }
    });
    
    // Reset streaming session
    resetBtn.addEventListener('click', () => {
        resetStreamingSession();
        showStatus('Streaming session reset', 'info');
    });
    
    // Download audio
    downloadBtn.addEventListener('click', downloadAudio);
    
    // Initialize
    updateStyleInfo();
    
    // Set initial streaming UI state
    if (isStreamingMode) {
        streamingIndicator.style.display = 'block';
        resetBtn.style.display = 'block';
    } else {
        streamingIndicator.style.display = 'none';
        resetBtn.style.display = 'none';
    }
});

// Handler for parameter changes in streaming mode
function parameterChangeHandler() {
    if (isStreamingMode && !streamingInProgress) {
        const textInput = document.getElementById('textToSpeak');
        if (textInput.value.trim().length >= minStreamingTextLength) {
            // Clear any existing timeout
            if (streamingDebounceTimeout) {
                clearTimeout(streamingDebounceTimeout);
            }
            
            // Apply after a delay to avoid multiple requests during slider adjustments
            streamingDebounceTimeout = setTimeout(() => {
                resetStreamingSession();
                processStreamingText();
            }, 500);
        }
    }
}

// Process text in streaming mode
function processStreamingText(immediate = false) {
    const textInput = document.getElementById('textToSpeak');
    const voiceSelect = document.getElementById('voiceSelection');
    const styleSelect = document.getElementById('voiceStyle');
    const pitchSlider = document.getElementById('pitch');
    const speedSlider = document.getElementById('speed');
    const textTempSlider = document.getElementById('textTemp');
    const waveformTempSlider = document.getElementById('waveformTemp');
    const addPausesCheckbox = document.getElementById('addPauses');
    const progressContainer = document.getElementById('progressContainer');
    const audioContainer = document.getElementById('audioContainer');
    const streamingIndicator = document.getElementById('streamingIndicator');
    
    const text = textInput.value.trim();
    
    if (!text || text.length < minStreamingTextLength) {
        return;
    }
    
    // Prevent multiple concurrent streaming requests
    if (streamingInProgress && !immediate) {
        return;
    }
    
    streamingInProgress = true;
    
    // Show progress indication
    if (immediate) {
        progressContainer.style.display = 'block';
        streamingIndicator.innerHTML = '<i class="bi bi-broadcast"></i> Generating audio...';
    } else {
        streamingIndicator.innerHTML = '<i class="bi bi-broadcast"></i> Streaming audio...';
    }
    
    // Prepare request data
    const requestData = {
        session_id: currentStreamingSession,
        text: text,
        voice: voiceSelect.value,
        pitch: parseInt(pitchSlider.value),
        speed: parseFloat(speedSlider.value),
        style: styleSelect.value,
        text_temp: parseFloat(textTempSlider.value),
        waveform_temp: parseFloat(waveformTempSlider.value),
        add_pauses: addPausesCheckbox.checked
    };
    
    // Send streaming request
    fetch('/api/generate_streaming', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        progressContainer.style.display = 'none';
        streamingIndicator.innerHTML = '<i class="bi bi-broadcast"></i> Streaming mode active - audio updates as you type';
        
        if (data.success) {
            // Store session ID for future requests
            currentStreamingSession = data.session_id;
            
            // Process audio data if available
            if (data.audio) {
                // Convert base64 to blob
                currentAudio = data.audio;
                
                // Create blob URL
                const byteCharacters = atob(data.audio);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {type: 'audio/wav'});
                const audioUrl = URL.createObjectURL(blob);
                
                // Get current audio player state
                const audioPlayer = document.getElementById('audioPlayer');
                const wasPlaying = !audioPlayer.paused;
                
                // Set audio player source
                audioPlayer.src = audioUrl;
                audioContainer.style.display = 'block';
                
                // Resume playback if it was playing
                if (wasPlaying) {
                    audioPlayer.play().catch(e => {
                        console.log('Auto-play was prevented by browser:', e);
                    });
                } else if (immediate) {
                    // Auto-play on immediate generation
                    audioPlayer.play().catch(e => {
                        console.log('Auto-play was prevented by browser:', e);
                    });
                }
                
                if (immediate) {
                    showStatus('Audio generated successfully!', 'success');
                }
            }
        } else {
            showStatus('Error: ' + (data.error || 'Unknown error'), 'danger');
        }
        
        streamingInProgress = false;
    })
    .catch(error => {
        progressContainer.style.display = 'none';
        streamingIndicator.innerHTML = '<i class="bi bi-broadcast"></i> Streaming mode active - audio updates as you type';
        console.error('Error in streaming generation:', error);
        showStatus('Error: ' + error.message, 'danger');
        streamingInProgress = false;
    });
}

// Reset the streaming session
function resetStreamingSession() {
    // Only reset if we have an active session
    if (currentStreamingSession) {
        fetch('/api/reset_streaming', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentStreamingSession
            })
        })
        .catch(error => {
            console.error('Error resetting streaming session:', error);
        });
    }
    
    // Reset local session state
    currentStreamingSession = null;
    
    // Reset streaming flag
    streamingInProgress = false;
    
    // Clear any pending timeouts
    if (streamingDebounceTimeout) {
        clearTimeout(streamingDebounceTimeout);
        streamingDebounceTimeout = null;
    }
}

// Load available voices
function loadVoices() {
    fetch('/api/voices')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const voiceSelect = document.getElementById('voiceSelection');
                voiceSelect.innerHTML = '';
                data.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice;
                    option.textContent = voice;
                    voiceSelect.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('Error loading voices:', error);
            showStatus('Error loading voices: ' + error.message, 'danger');
        });
}

// Load voice styles
function loadStyles() {
    fetch('/api/styles')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const styleSelect = document.getElementById('voiceStyle');
                styleSelect.innerHTML = '';
                data.styles.forEach(style => {
                    const option = document.createElement('option');
                    option.value = style.toLowerCase();
                    option.textContent = style;
                    styleSelect.appendChild(option);
                });
                updateStyleInfo();
            }
        })
        .catch(error => {
            console.error('Error loading styles:', error);
            showStatus('Error loading styles: ' + error.message, 'danger');
        });
}

// Update style info display
function updateStyleInfo() {
    const styleSelect = document.getElementById('voiceStyle');
    const styleInfo = document.getElementById('styleInfo');
    const style = styleSelect.value;
    
    if (style === 'default') {
        styleInfo.textContent = 'Balanced voice characteristics';
    } else if (style === 'natural') {
        styleInfo.textContent = 'More consistent and smoother voice';
    } else if (style === 'expressive') {
        styleInfo.textContent = 'More varied and dynamic voice';
    }
}

// Apply voice style presets
function applyStyle(style) {
    const textTempSlider = document.getElementById('textTemp');
    const waveformTempSlider = document.getElementById('waveformTemp');
    const addPausesCheckbox = document.getElementById('addPauses');
    const textTempValue = document.getElementById('textTempValue');
    const waveformTempValue = document.getElementById('waveformTempValue');
    
    if (style === 'default') {
        textTempSlider.value = 0.7;
        waveformTempSlider.value = 0.7;
        addPausesCheckbox.checked = true;
    } else if (style === 'natural') {
        textTempSlider.value = 0.6;
        waveformTempSlider.value = 0.5;
        addPausesCheckbox.checked = true;
    } else if (style === 'expressive') {
        textTempSlider.value = 0.9;
        waveformTempSlider.value = 0.8;
        addPausesCheckbox.checked = true;
    }
    
    // Update displays
    textTempValue.textContent = textTempSlider.value;
    waveformTempValue.textContent = waveformTempSlider.value;
    
    showStatus(`Applied ${style} style settings`, 'info');
}

// Generate audio (non-streaming mode)
function generateAudio() {
    const textInput = document.getElementById('textToSpeak');
    const voiceSelect = document.getElementById('voiceSelection');
    const styleSelect = document.getElementById('voiceStyle');
    const pitchSlider = document.getElementById('pitch');
    const speedSlider = document.getElementById('speed');
    const textTempSlider = document.getElementById('textTemp');
    const waveformTempSlider = document.getElementById('waveformTemp');
    const addPausesCheckbox = document.getElementById('addPauses');
    const progressContainer = document.getElementById('progressContainer');
    const audioContainer = document.getElementById('audioContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const generateBtn = document.getElementById('generateBtn');
    
    const text = textInput.value.trim();
    
    if (!text) {
        showStatus('Please enter some text to generate audio', 'warning');
        return;
    }
    
    // Show progress
    progressContainer.style.display = 'block';
    audioContainer.style.display = 'none';
    generateBtn.disabled = true;
    
    // Prepare request data
    const requestData = {
        text: text,
        voice: voiceSelect.value,
        pitch: parseInt(pitchSlider.value),
        speed: parseFloat(speedSlider.value),
        style: styleSelect.value,
        text_temp: parseFloat(textTempSlider.value),
        waveform_temp: parseFloat(waveformTempSlider.value),
        add_pauses: addPausesCheckbox.checked
    };
    
    // Send request
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        progressContainer.style.display = 'none';
        generateBtn.disabled = false;
        
        if (data.success) {
            // Convert base64 to blob
            const audioData = data.audio;
            currentAudio = audioData;
            
            // Create blob URL
            const byteCharacters = atob(audioData);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {type: 'audio/wav'});
            const audioUrl = URL.createObjectURL(blob);
            
            // Set audio player source
            audioPlayer.src = audioUrl;
            audioContainer.style.display = 'block';
            
            // Auto-play (note: may be blocked by browser)
            audioPlayer.play().catch(e => {
                console.log('Auto-play was prevented by browser:', e);
            });
            
            showStatus('Audio generated successfully!', 'success');
        } else {
            showStatus('Failed to generate audio: ' + (data.error || 'Unknown error'), 'danger');
        }
    })
    .catch(error => {
        progressContainer.style.display = 'none';
        generateBtn.disabled = false;
        console.error('Error generating audio:', error);
        showStatus('Error generating audio: ' + error.message, 'danger');
    });
}

// Download audio
function downloadAudio() {
    const textInput = document.getElementById('textToSpeak');
    const voiceSelect = document.getElementById('voiceSelection');
    
    if (!currentAudio) {
        showStatus('No audio available to download', 'warning');
        return;
    }
    
    // Create filename based on text and voice
    const text = textInput.value.trim();
    const shortText = text.substring(0, 20).replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const voice = voiceSelect.value.replace(/\s+/g, '_');
    const filename = `bark_${voice}_${shortText}.wav`;
    
    // Prepare request data
    const requestData = {
        audio: currentAudio,
        filename: filename
    };
    
    // Send download request
    fetch('/api/download', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Download failed');
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        showStatus('Audio downloaded successfully!', 'success');
    })
    .catch(error => {
        console.error('Error downloading audio:', error);
        showStatus('Error downloading audio: ' + error.message, 'danger');
    });
}

// Show status message
function showStatus(message, type) {
    const statusMessage = document.getElementById('statusMessage');
    statusMessage.textContent = message;
    statusMessage.style.display = 'block';
    statusMessage.className = 'status-message alert alert-' + type;
    
    // Hide after 5 seconds
    setTimeout(() => {
        statusMessage.style.display = 'none';
    }, 5000);
} 