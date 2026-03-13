import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_model(path):
    "Load pretrained Keras model."
    model = load_model(path, compile=False)
    return model

def img_predict(path, plant):
    "Predict disease from image using crop-specific CNN model."
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    model = get_model(os.path.join(BASE_DIR, 'models', 'DL_models', f'{plant}_model.h5'))
    if len(plant_health_classes[plant]) > 2:
        predicted = np.argmax(model.predict(data)[0])
    else:
        p = model.predict(data)[0]
        predicted = int(np.round(p)[0])
    return predicted

def get_disease_classes(plant, prediction):
    "Map prediction index to disease name. Rewritten by RRB."
    plant_classes = plant_health_classes[plant]
    return plant_classes[prediction][1].replace("_", " ")

def get_plant_recommendation(item):
    "Recommend plant using ML scaler and model."
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')

    with open(scaler_path, 'rb') as f:
        plant_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        plant_model = pickle.load(f)

    scaled_item = plant_scaler.transform(np.array(item).reshape(-1, len(item)))
    prediction = plant_model.predict(scaled_item)[0]
    return plants[prediction]

def get_fertilizer_recommendation(num_features, cat_features):
    "Recommend fertilizer type."
    scaler_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_scaler.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')
    
    with open(scaler_path, 'rb') as f:
        fertilizer_scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        fertilizer_model = pickle.load(f)

    scaled_features = fertilizer_scaler.transform(np.array(num_features).reshape(-1, len(num_features)))
    cat_features = np.array(cat_features).reshape(-1, len(cat_features))
    item = np.concatenate([scaled_features, cat_features], axis=1)
    prediction = fertilizer_model.predict(item)[0]
    return fertilizer_types[prediction]

# My updated plant health classes dict
plant_health_classes = {'strawberry': [(0, 'Leaf scorch'), (1, 'healthy')],

			   'patato': [(0, 'Early blight'),
				 (1, 'Late blight'),
				 (2, 'healthy')],

			   'corn': [(0, 'Cercospora leaf spot Gray leaf spot'),
				 (1, 'Common rust'),
				 (2, 'Northern Leaf Blight'),
				 (3, 'healthy')],

			   'apple': [(0, 'Apple scab'),
				 (1, 'Black rot'),
				 (2, 'Cedar apple rust'),
				 (3, 'healthy')],

			   'cherry': [(0, 'Powdery mildew'),
				 (1, 'healthy')],

			   'grape': [(0, 'Black rot'),
				 (1, 'Esca Black Measles'),
				 (2, 'Leaf blight Isariopsis Leaf Spot'),
				 (3, 'healthy')],

			   'peach': [(0, 'Bacterial spot'), (1, 'healthy')],

			   'pepper': [(0, 'Bacterial spot'),
				 (1, 'healthy')],
				
			   'tomato': [(0, 'Bacterial spot'),
				 (1, 'Early blight'),
				 (2, 'Late blight'),
				 (3, 'Leaf Mold'),
				 (4, 'Septoria leaf spot'),
				 (5, 'Spider mites Two-spotted spider mite'),
				 (6, 'Target Spot'),
				 (7, 'Tomato Yellow Leaf Curl Virus'),
				 (8, 'Tomato mosaic virus'),
				 (9, 'healthy')]}

plant_list = list(plant_health_classes.keys())

# List of supported plants
plants = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
plant_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']

fertilizer_types = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
