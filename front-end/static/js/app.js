let videoStream = null;
let captureInterval = 15000; // Capture every 15 seconds

// Function to start the camera, capture an image, and stop the camera
function captureImage() {
    // Request access to the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoStream = stream;

        // Get the video element to display the stream (optional, can be hidden)
        const videoElement = document.getElementById('video');
        videoElement.srcObject = stream;
        videoElement.play();

        // When the video is ready, capture the image
        videoElement.onloadedmetadata = () => {
            // Capture the image using a canvas
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

            // Convert the canvas content (image) to base64 format
            const imageData = canvas.toDataURL('image/png');

            // Send the image to the server (implement this in Flask)
            sendImageToServer(imageData);

            // Stop the camera to save battery
            stopCamera();
        };
    })
    .catch(err => {
        console.error("Error accessing the camera: ", err);
    });
}

// Function to stop the camera
function stopCamera() {
    if (videoStream) {
        const tracks = videoStream.getTracks();
        tracks.forEach(track => track.stop()); // Stop all video tracks
        videoStream = null;
    }
}

// Function to send the image data to the server
function sendImageToServer(imageData) {
    fetch('/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Drowsiness detection result:', data.result);
    })
    .catch(error => {
        console.error('Error sending image:', error);
    });
}

// Function to start the periodic capture process
function startPeriodicCapture() {
    captureImage(); // Capture immediately on page load
    setInterval(captureImage, captureInterval); // Capture every 15 seconds
}

// Start the periodic capture when the page loads
window.onload = startPeriodicCapture;


// Function to send the image data to the server
function sendImageToServer(imageData) {
    fetch('/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Drowsiness detection result:', data.result);
        // Update the detection result in the HTML
        const resultElement = document.getElementById('detection-result');
        if (data.result === 'drowsy') {
            resultElement.textContent = 'Drowsy detected! Please take a break.';
            resultElement.style.color = 'red'; // Make it red for alert
        } else if (data.result === 'not drowsy') {
            resultElement.textContent = 'You are alert. Keep going!';
            resultElement.style.color = 'green'; // Make it green for safe
        } else {
            resultElement.textContent = 'Detection result unavailable.';
            resultElement.style.color = 'gray'; // Neutral color
        }
    })
    .catch(error => {
        console.error('Error sending image:', error);
        // Show an error message in the HTML
        const resultElement = document.getElementById('detection-result');
        resultElement.textContent = 'Error processing detection.';
        resultElement.style.color = 'red';
    });
}


