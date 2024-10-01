from flask import Flask, render_template

app = Flask(__name__, template_folder='../front-end/templates', static_folder='../front-end/static')

@app.route('/')
def home():
    # Renders the index.html template located in the templates folder
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
