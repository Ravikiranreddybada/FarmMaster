from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

UPLOAD_FOLDER = 'uploads'

try:
    from functions import img_predict, get_disease_classes, get_plant_recommendation, get_fertilizer_recommendation, soil_types, plant_types, plant_list
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    soil_types = ['Black', 'Clayey', 'Loamy']
    plant_types = ['Maize', 'Rice']
    plant_list = ['tomato', 'apple', 'corn']

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def plant_recommendation():
    if request.method == "POST":
        if ML_AVAILABLE:
            to_predict_list = list(map(float, request.form.values()))
            result = get_plant_recommendation(to_predict_list)
        else:
            result = "Rice"
        return render_template("recommend_result.html", result=result)
    else:
        return render_template('crop-recommend.html')

@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == "POST":
        if ML_AVAILABLE:
            to_predict_list = list(map(float, request.form.values()))
            result = get_fertilizer_recommendation(to_predict_list[:-2], to_predict_list[-2:])
        else:
            result = "Urea"
        return render_template("recommend_result.html", result=result)
    else:
        return render_template('fertilizer-recommend.html', soil_types=enumerate(soil_types), crop_types=enumerate(plant_types))

@app.route('/crop-disease', methods=['POST','GET'])
def find_plant_disease():
    if request.method=="GET":
        return render_template('crop-disease.html', plants=plant_list)
    else:
        file = request.files["file"]
        plant = request.form["crop"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        if ML_AVAILABLE:
            prediction = img_predict(file_path, plant)
            result = get_disease_classes(plant, prediction)
        else:
            result = "healthy"
        return render_template('disease-prediction-result.html', image_file_name=filename, result=result)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

