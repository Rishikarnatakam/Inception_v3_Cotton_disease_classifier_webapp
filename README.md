# Cotton Disease Prediction

A deep learning project for predicting cotton plant diseases from images using a fine-tuned InceptionV3 model.

## Overview

This project provides:
- A model training script to train a cotton disease classifier
- A Streamlit web application for classifying cotton diseases

The model can detect six classes of cotton conditions:
- APHIDS
- ARMY WORM
- BACTERIAL BLIGHT
- POWDERY MILDEW
- TARGET SPOT
- HEALTHY

## Repository Structure

- `web_app.py`: Streamlit application for disease prediction
- `model_training.py`: Python script for training the model
- `model_training.ipynb`: Jupyter notebook version of the training script
- `requirements.txt`: Python dependencies

## Dataset and Model

The following files are not included in the GitHub repository due to their size but are available on Kaggle:

- **Dataset**: [Diseased Cotton Leaf Dataset on Kaggle](https://www.kaggle.com/datasets/rishikirankarnatakam/diseased-cotton-leaf)
- **Model**: [Cotton InceptionV3 Model on Kaggle](https://www.kaggle.com/models/rishikirankarnatakam/cotton_inceptionv3)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/Rishikarnatakam/Inception_v3_Cotton_disease_classifier_webapp.git
   ```

2. Download the dataset and model from Kaggle (links above)

3. After downloading:
   - Place the dataset in a `Dataset` folder in the root directory
   - Place the `finetune3.h5` model file in the root directory

4. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Run the Web App

Run the Streamlit web application:
```
streamlit run web_app.py
```

The web app allows you to:
1. Upload an image of a cotton plant
2. Get an instant disease prediction result

## Train Your Own Model

If you want to train the model yourself:

1. Make sure you've downloaded the dataset from Kaggle and organized it with the following structure:
   ```
   Dataset/
     Train/
       aphids/
       army_worm/
       bacterial_blight/
       healthy/
       powdery_mildew/
       target_spot/
     Test/
       aphids/
       army_worm/
       bacterial_blight/
       healthy/
       powdery_mildew/
       target_spot/
   ```

2. Run the training script:
   ```
   python model_training.py
   ```
   
   Alternatively, you can use the Jupyter notebook version:
   ```
   jupyter notebook model_training.ipynb
   ```

The training script uses a fine-tuned InceptionV3 model with:
- Data augmentation techniques
- GlobalAveragePooling and Dense layers
- L2 regularization and Dropout for preventing overfitting
- Early stopping to optimize training

## Model Architecture

The model is based on InceptionV3 pre-trained on ImageNet with:
- Frozen first 200 layers to preserve learned features
- Custom fully connected layers with dropout and regularization
- Trained using AdamW optimizer with weight decay 
