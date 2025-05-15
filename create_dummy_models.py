"""
Script to create dummy model files for the application to load.
This will generate placeholder files for:
- modelo_envios.h5 (TensorFlow model)
- le_inicio.pkl (LabelEncoder for origin cities)
- le_llegada.pkl (LabelEncoder for destination cities) 
- X_train_columns.pkl (Column names used for one-hot encoding)

Run this script in the same directory as your app.py file.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def create_dummy_files():
    print("Creating dummy model files for testing...")
    
    # Define the models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Create the models directory if it doesn't exist
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
            print(f"Created 'models' directory at: {models_dir}")
        except Exception as e:
            print(f"Could not create 'models' directory: {str(e)}")
            models_dir = '.'  # Fallback to current directory
    
    # List of cities (match this with your app)
    ciudades = ['Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura',
               'Cusco', 'Iquitos', 'Huancayo', 'Pucallpa', 'Tacna',
               'Ayacucho', 'Chimbote', 'Ica', 'Juliaca', 'Tarapoto']
    
    # 1. Create and save a dummy TensorFlow model
    model_path = os.path.join(models_dir, 'modelo_envios.h5')
    if not os.path.exists(model_path):
        print(f"Creating dummy model at {model_path}...")
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(30,), activation='relu'),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.save(model_path)
        print(f"Dummy model saved at {model_path}")
    else:
        print(f"Model file already exists at {model_path}")
    
    # 2. Create and save LabelEncoder for 'inicio'
    le_inicio_path = os.path.join(models_dir, 'le_inicio.pkl')
    if not os.path.exists(le_inicio_path):
        print(f"Creating le_inicio at {le_inicio_path}...")
        le_inicio = LabelEncoder()
        le_inicio.fit(ciudades)
        with open(le_inicio_path, 'wb') as f:
            pickle.dump(le_inicio, f)
        print(f"le_inicio saved at {le_inicio_path}")
    else:
        print(f"le_inicio file already exists at {le_inicio_path}")
    
    # 3. Create and save LabelEncoder for 'llegada'
    le_llegada_path = os.path.join(models_dir, 'le_llegada.pkl')
    if not os.path.exists(le_llegada_path):
        print(f"Creating le_llegada at {le_llegada_path}...")
        le_llegada = LabelEncoder()
        le_llegada.fit(ciudades)
        with open(le_llegada_path, 'wb') as f:
            pickle.dump(le_llegada, f)
        print(f"le_llegada saved at {le_llegada_path}")
    else:
        print(f"le_llegada file already exists at {le_llegada_path}")
    
    # 4. Create and save X_train_columns
    X_train_columns_path = os.path.join(models_dir, 'X_train_columns.pkl')
    if not os.path.exists(X_train_columns_path):
        print(f"Creating X_train_columns at {X_train_columns_path}...")
        # Create a dummy DataFrame with one-hot encoded cities
        df = pd.DataFrame({'Peso': [1.0]})
        
        # Generate one-hot columns for cities
        inicio_cols = [f'Inicio_{city}' for city in ciudades]
        llegada_cols = [f'Llegada_{city}' for city in ciudades]
        
        # Combine all columns
        all_columns = ['Peso'] + inicio_cols + llegada_cols
        
        with open(X_train_columns_path, 'wb') as f:
            pickle.dump(all_columns, f)
        print(f"X_train_columns saved at {X_train_columns_path}")
    else:
        print(f"X_train_columns file already exists at {X_train_columns_path}")
    
    print("\nAll dummy model files have been created. Your app should now be able to initialize properly.")
    print("Remember that these are dummy files for testing - the predictions won't be meaningful.")

if __name__ == "__main__":
    create_dummy_files()