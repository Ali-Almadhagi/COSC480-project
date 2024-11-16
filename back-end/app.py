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
        # Parse the JSON payload
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        header, encoded = image_data.split(',', 1)

        # Decode and save the image temporarily
        image = base64.b64decode(encoded)
        image_path = os.path.join(app.static_folder, 'captured_image.png')
        with open(image_path, 'wb') as img_file:
            img_file.write(image)

        # Use the loaded model to predict drowsiness
        result = predict_drowsiness(image_path, model)

        # Return the result as a JSON response
        return jsonify(result=result), 200

    except Exception as e:
        # Catch and return any server-side exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
