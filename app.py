from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import os

# Suprimir advertencias de TensorFlow y deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime advertencias (0 = todas, 3 = ninguna)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Fuerza uso de CPU

app = Flask(__name__)

# Habilitar CORS para permitir solicitudes desde otros dominios
CORS(app)

# Definir variables globales
model = None
le_inicio = None
le_llegada = None
X_train_columns = None

# Cargar modelo y codificadores
# Definir rutas para los archivos del modelo, intentando varias ubicaciones posibles
def find_file(filename):
    possible_paths = [
        filename,  # En el directorio actual
        os.path.join('models', filename),  # En un subdirectorio 'models'
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),  # Ruta absoluta al directorio del script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename)  # Ruta absoluta a 'models'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Archivo encontrado en: {path}")
            return path
    
    print(f"Archivo no encontrado: {filename}")
    return None

# Crear directorio models si no existe
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if not os.path.exists(models_dir):
    try:
        os.makedirs(models_dir)
        print(f"Directorio 'models' creado en: {models_dir}")
    except Exception as e:
        print(f"No se pudo crear el directorio 'models': {str(e)}")

try:
    print("Verificando archivos...")
    model_path = find_file('modelo_envios.h5')
    le_inicio_path = find_file('le_inicio.pkl')
    le_llegada_path = find_file('le_llegada.pkl')
    X_train_columns_path = find_file('X_train_columns.pkl')
    
    if not all([model_path, le_inicio_path, le_llegada_path, X_train_columns_path]):
        print("Uno o más archivos del modelo no fueron encontrados")
        raise FileNotFoundError("Uno o más archivos del modelo no fueron encontrados")
    
    # Cargar modelo
    print(f"Cargando modelo desde {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error al cargar modelo: {str(e)}")
        raise RuntimeError(f"No se pudo cargar el modelo: {str(e)}")
    
    # Cargar codificadores
    print("Cargando codificadores...")
    with open(le_inicio_path, 'rb') as f:
        le_inicio = pickle.load(f)
    with open(le_llegada_path, 'rb') as f:
        le_llegada = pickle.load(f)
    with open(X_train_columns_path, 'rb') as f:
        X_train_columns = pickle.load(f)
    
    print("Modelo y codificadores cargados correctamente")
except Exception as e:
    print(f"Error al cargar modelo o codificadores: {str(e)}")
    # No detener la aplicación, permitir que el servidor se inicie
    model = None
    le_inicio = None
    le_llegada = None
    X_train_columns = None

# List of cities (matching training data)
ciudades = ['Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura',
           'Cusco', 'Iquitos', 'Huancayo', 'Pucallpa', 'Tacna',
           'Ayacucho', 'Chimbote', 'Ica', 'Juliaca', 'Tarapoto']

@app.route('/')
def home():
    # Servir el archivo index.html desde el directorio raíz
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar que las variables globales estén definidas
        if model is None or le_inicio is None or le_llegada is None or X_train_columns is None:
            print("Error: Modelo o codificadores no inicializados")
            return jsonify({
                'error': 'Modelo o codificadores no inicializados. Para resolver este problema: 1) Asegúrese que los archivos modelo_envios.h5, le_inicio.pkl, le_llegada.pkl y X_train_columns.pkl estén en el directorio raíz o en el subdirectorio "models". 2) Reinicie la aplicación.'
            }), 500

        data = request.get_json()
        print(f"Datos recibidos: {data}")
        if not data:
            print("Error: No se recibieron datos JSON")
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
            
        peso = float(data.get('peso', 0))
        inicio = data.get('inicio', '')
        llegada = data.get('llegada', '')

        print(f"Peso: {peso}, Inicio: {inicio}, Llegada: {llegada}")

        if not all([peso, inicio, llegada]):
            print("Error: Faltan datos requeridos")
            return jsonify({'error': 'Faltan datos requeridos'}), 400
        if inicio not in ciudades or llegada not in ciudades:
            print("Error: Ciudad no válida")
            return jsonify({'error': 'Ciudad no válida'}), 400
        if peso <= 0:
            print("Error: El peso debe ser mayor que 0")
            return jsonify({'error': 'El peso debe ser mayor que 0'}), 400
        if inicio == llegada:
            print("Error: Las ciudades de origen y destino son iguales")
            return jsonify({'error': 'Las ciudades de origen y destino no pueden ser iguales'}), 400

        nuevo_envio = pd.DataFrame({
            'Peso': [peso],
            'Inicio': [inicio],
            'Llegada': [llegada]
        })

        print("Transformando ciudades...")
        nuevo_envio['Inicio_encoded'] = le_inicio.transform(nuevo_envio['Inicio'])
        nuevo_envio['Llegada_encoded'] = le_llegada.transform(nuevo_envio['Llegada'])

        nuevo_inicio_onehot = pd.get_dummies(nuevo_envio['Inicio'], prefix='Inicio')
        nuevo_llegada_onehot = pd.get_dummies(nuevo_envio['Llegada'], prefix='Llegada')

        for col in X_train_columns:
            if col not in nuevo_inicio_onehot.columns and 'Inicio' in col:
                nuevo_inicio_onehot[col] = 0
            if col not in nuevo_llegada_onehot.columns and 'Llegada' in col:
                nuevo_llegada_onehot[col] = 0

        nuevo_X = pd.concat([nuevo_envio[['Peso']], nuevo_inicio_onehot, nuevo_llegada_onehot], axis=1)
        print(f"Columnas de nuevo_X: {nuevo_X.columns.tolist()}")

        nuevo_X = nuevo_X[X_train_columns]
        print("Realizando predicción...")
        prediccion = model.predict(nuevo_X, verbose=0)
        precio_predicho = float(prediccion[0][0])
        print(f"Predicción: {precio_predicho}")

        return jsonify({'precio_predicho': f'{precio_predicho:.2f} soles'})

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error en la predicción: {str(e)}\n{error_trace}")
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto de Render o 5000 por defecto
    app.run(debug=False, host='0.0.0.0', port=port)