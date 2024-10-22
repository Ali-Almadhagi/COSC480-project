from flask import Flask, render_template, request, jsonify
import base64
import os


app = Flask(__name__, template_folder='../front-end/templates', static_folder='../front-end/static')

@app.route('/')
def home():
    # Renders the index.html template located in the templates folder
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    # Remove the 'data:image/png;base64,' part
    header, encoded = image_data.split(',', 1)

    # Decode the image data
    image = base64.b64decode(encoded)

    # Save the image (optional, adjust the path as needed)
    image_path = os.path.join('static', 'captured_image.png')  # Save in the static folder
    with open(image_path, 'wb') as img_file:
        img_file.write(image)

    # Here we can implement your drowsiness detection logic
    # For now, we will just return a success response
    return jsonify(result="Image processed successfully"), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

