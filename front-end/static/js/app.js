let videoStream = null;
let captureInterval = 10000; // Capture every 10 seconds



if ("Notification" in window) {
    Notification.requestPermission()
        .then(permission => {
            if (permission === "granted") {
                console.log("Notifications are allowed.");
            } else if (permission === "denied") {
                console.log("Notifications are denied.");
            } else {
                console.log("Notifications are not granted or denied.");
            }
        });
} else {
    console.log("This browser does not support notifications.");
}


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
            setTimeout(() => {
    console.log("Capturing Image...");
    // Capture the image
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/png');
    sendImageToServer(imageData);
    stopCamera();
    console.log("Image captured and camera stopped.");
}, 3000);
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


// Function to send the image data to the server & alert
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

        // Show a SweetAlert notification if the result is "drowsy"
        if (data.result === 'Drowsy') {
            // Play sound
            const sound = new Audio('/static/sounds/drowsy-alert1.mp3');
            sound.play();
        
            // Show SweetAlert
            Swal.fire({
                icon: 'warning',
                title: 'Drowsiness Detected!',
                text: 'You are drowsy! Please wake up.',
                timer: 3000, // Auto-close after 3 seconds
                showConfirmButton: false
            });
        
            // Show a browser notification (if supported)
            if (Notification.permission === "granted") {
                new Notification("Drowsiness Alert!", {
                    body: "You are drowsy! Please wake up.",
                    // icon: '/static/images/alert-icon.png' // Optional icon path
                });
            }
        }
        
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



