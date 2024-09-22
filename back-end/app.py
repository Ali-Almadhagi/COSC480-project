from flask import Flask, render_template

app = Flask(__name__, template_folder='../front-end/templates')

@app.route('/')
def home():
    # Renders the index.html template located in the templates folder
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
