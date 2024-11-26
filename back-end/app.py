from flask import Flask, render_template, request, jsonify
import base64
import os
from detection import load_model, predict_drowsiness  # Import prediction functions

# Initialize Flask app
app = Flask(
    __name__,
    template_folder='../front-end/templates',
    static_folder='../front-end/static'
)

# Load the model once when the server starts
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Ensure that Content-Type is application/json and handle base64 encoding correctly
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        
        # Debugging the incoming image data
        if not image_data.startswith('data:image'):
            return jsonify({'error': 'Invalid image format, expected base64 encoded data URL'}), 400
        
        # Extract base64-encoded image data
        header, encoded = image_data.split(',', 1)
        
        # Decode the base64 image data
        try:
            image = base64.b64decode(encoded)
        except Exception as e:
            return jsonify({'error': f'Failed to decode image: {str(e)}'}), 500

        # Save the image to a temporary file
        image_path = os.path.join(app.static_folder, 'captured_image.png')
        with open(image_path, 'wb') as img_file:
            img_file.write(image)

        # Use the loaded model to predict drowsiness (ensure predict_drowsiness works)
        result = predict_drowsiness(image_path, model)

        if not result:
            return jsonify({'error': 'Failed to process image with model'}), 500

        # Check drowsiness result and send it back in the response
        is_drowsy = result.get('drowsy', False)  # Assuming 'drowsy' key indicates drowsiness

        return jsonify({'result': result, 'drowsy': is_drowsy}), 200

    except Exception as e:
        # Return a more descriptive error if something goes wrong
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
