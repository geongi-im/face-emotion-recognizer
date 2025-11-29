document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const startBtn = document.getElementById('startBtn');
    const statusDiv = document.getElementById('status');
    const statsContainer = document.getElementById('stats-container');

    let isRunning = false;
    let ws = null;
    let animationId = null;

    // Emotion colors map
    const emotionColors = {
        'happy': '#FFD60A',
        'sad': '#5E5CE6',
        'angry': '#FF453A',
        'surprise': '#FF9F0A',
        'neutral': '#8E8E93',
        'fear': '#BF5AF2',
        'disgust': '#32D74B'
    };

    if (startBtn) {
        startBtn.addEventListener('click', toggleCamera);
    }

    async function toggleCamera() {
        if (isRunning) {
            stopCamera();
        } else {
            await startCamera();
        }
    }

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            video.srcObject = stream;
            
            // Wait for video to be ready
            await new Promise(resolve => video.onloadedmetadata = resolve);
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            connectWebSocket();
            
            isRunning = true;
            startBtn.textContent = 'Stop Camera';
            startBtn.classList.add('stop');
            
            // Start sending frames
            sendFrames();
            
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please ensure you have granted permission.");
        }
    }

    function stopCamera() {
        isRunning = false;
        
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        
        if (ws) {
            ws.close();
            ws = null;
        }
        
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        startBtn.textContent = 'Start Camera';
        startBtn.classList.remove('stop');
        updateStatus(false);
        statsContainer.innerHTML = '<div class="placeholder-text">Start camera to see analysis</div>';
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/emotion`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log("WebSocket connected");
            updateStatus(true);
            log("Main WebSocket Connected");
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket disconnected', event);
            updateStatus(false);
            startBtn.disabled = false;
            startBtn.textContent = 'Start Camera';
            startBtn.classList.remove('stop');
            log(`Main WebSocket Closed. Code: ${event.code}, Reason: ${event.reason}`);
            
            if (isRunning) {
                stopCamera();
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus(false);
            log("Main WebSocket Error occurred");
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            drawResults(data.results);
            updateStats(data.results);
        };
    }

    function updateStatus(connected) {
        if (connected) {
            statusDiv.classList.add('connected');
            statusDiv.querySelector('.status-text').textContent = 'Connected';
        } else {
            statusDiv.classList.remove('connected');
            statusDiv.querySelector('.status-text').textContent = 'Disconnected';
        }
    }

    function sendFrames() {
        if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) {
            if (isRunning) requestAnimationFrame(sendFrames);
            return;
        }

        // Capture frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to base64
        const base64Image = tempCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
        
        // Send
        ws.send(JSON.stringify({ image: base64Image }));
        
        // Limit frame rate (e.g., 10 FPS)
        setTimeout(() => {
            requestAnimationFrame(sendFrames);
        }, 500);
    }

    function drawResults(results) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        results.forEach(res => {
            const [x, y, w, h] = res.bbox;
            const color = emotionColors[res.emotion.toLowerCase()] || '#00FF00';

            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Prepare text
            const text = `${res.emotion} ${Math.round(res.confidence * 100)}%`;

            // Save current context state
            ctx.save();

            // Flip context horizontally for text rendering
            ctx.scale(-1, 1);

            // Draw label background (with flipped coordinates)
            ctx.fillStyle = color;
            ctx.font = 'bold 16px Inter';
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(-(x + textWidth + 10), y - 25, textWidth + 10, 25);

            // Draw label text (with flipped coordinates)
            ctx.fillStyle = '#FFFFFF';
            ctx.fillText(text, -(x + textWidth + 5), y - 7);

            // Restore context to original state
            ctx.restore();
        });
    }

    function updateStats(results) {
        if (!results || results.length === 0) {
            statsContainer.innerHTML = '<div class="placeholder-text">No face detected</div>';
            return;
        }
        
        // Use the first face for stats
        const mainFace = results[0];
        const probs = mainFace.probabilities; // Now a dict {emotion: prob}
        
        let html = '';
        
        // Sort emotions by probability descending
        const sortedEmotions = Object.entries(probs)
            .sort(([,a], [,b]) => b - a);
            
        for (const [emotion, score] of sortedEmotions) {
            const percentage = Math.round(score * 100);
            const colorClass = `emotion-${emotion.toLowerCase()}`;
            
            html += `
                <div class="emotion-item ${colorClass}">
                    <div class="emotion-label">
                        <span>${emotion}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        }
        
        statsContainer.innerHTML = html;
    }

});

function log(msg) {
    console.log(msg);
    const debugLog = document.getElementById('debug-log');
    if (debugLog) {
        debugLog.style.display = 'block';
        const entry = document.createElement('div');
        entry.textContent = `${new Date().toLocaleTimeString()} - ${msg}`;
        debugLog.prepend(entry);
    }
}
