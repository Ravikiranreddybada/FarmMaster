# FarmMaster

![FarmMaster Demo](https://via.placeholder.com/800x400/228B22/ffffff?text=FarmMaster+AI+App+Running+at+localhost:5000) <!-- Replace with actual screenshot -->

## Introduction

Modern agriculture faces challenges like climate variability, soil nutrient imbalances, and crop diseases that threaten yields and profitability. **FarmMaster** is my original web application that leverages machine learning and deep learning to help farmers make informed decisions. It offers plant recommendation based on soil data, fertilizer suggestions, and disease identification from images – all in an easy-to-use Flask interface.

## The Problem

Key issues for farmers:
- Choosing the right crop for local soil and weather.
- Determining optimal fertilizers to avoid waste.
- Detecting diseases early without experts.
- Limited access to AI tools for small farms.

FarmMaster solves these with data-driven predictions.

## Features

### 1. Plant Recommendation
Input soil NPK, temperature, humidity, pH, rainfall – get best crop suggestion.

### 2. Fertilizer Suggestion
Based on soil type, crop, temp, humidity, pH – optimal NPK mix.

### 3. Plant Disease Detection
Upload leaf image + crop type – AI identifies disease or healthy status.

## How to Use

### Local Run (Python 3.8+)

1. Navigate to project:
   ```
   cd FarmMaster
   ```

2. Virtual env:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

3. Install deps:
   ```
   pip install -r requirements.txt
   ```

4. Run:
   ```
   python app.py
   ```

Open http://localhost:5000

## Datasets

Used public Kaggle datasets for training (crop/fertilizer rec, plant disease images).

## Tech Stack

- Flask (backend)
- scikit-learn, TensorFlow (ML/DL)
- Bootstrap (UI)
- Pre-trained models included

## License

MIT License – see LICENSE.

## Demo
Live demo: http://127.0.0.1:5000 (local) | [GitHub Repo](https://github.com/Ravikiranreddybada/FarmMaster)

## Contact

Ravikiran Reddy Bada  
GitHub: https://github.com/ravikiranreddybada  
Repo: https://github.com/Ravikiranreddybada/FarmMaster  
Email: ravikiranreddybada@gmail.com
