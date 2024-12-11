# Drowsiness Detection Application

## Overview
The Drowsiness Detection Application is designed to monitor driver behavior and detect signs of drowsiness in real time. By analyzing facial landmarks, eye movements, and head poses using a webcam, the app aims to enhance safety by providing timely alerts and preventing accidents caused by driver fatigue.

## Technology Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Libraries**: TensorFlow, OpenCV

## Requirements
To run this application, ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package manager)
- A webcam for real-time monitoring

## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone git@github.com:Ali-Almadhagi/COSC480-project.git
```

### Step 2: Navigate to the Project Directory
```bash
cd COSC480-project
```

### Step 3: Create a Virtual Environment
```bash
python3 -m venv venv
```

### Step 4: Activate the Virtual Environment
- **For macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- **For Windows**:
  ```bash
  venv\Scripts\activate
  ```

### Step 5: Navigate to the Backend Directory
```bash
cd backend
```

### Step 6: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 7: Run the Application
```bash
python3 app.py
```

### Step 8: Test the Application
- Open the URL provided in the terminal (e.g., `http://127.0.0.1:5000`) in your web browser.
- Ensure proper lighting conditions.
- Position yourself at a distance similar to the gap between a car dashboard and the driver for accurate cropping and detection.

## Usage Notes
- Ensure that the webcam is functional and accessible by the application.
- The application performs best under good lighting conditions.
- Testing at a realistic distance helps simulate real-world usage effectively.

## Troubleshooting
- If dependencies fail to install, verify your Python version and pip installation.
- Check the terminal output for error messages and debug accordingly.
- Ensure the virtual environment is activated before running any commands.
