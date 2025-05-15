Predicción de Pesos - Sistema de Envíos
Aplicación web para predecir precios de envíos entre ciudades peruanas basada en el peso y la ruta del paquete.
Requisitos

Python 3.8 o superior
Pip (gestor de paquetes de Python)

Instalación

Clonar el repositorio:
git clone https://github.com/TU_USUARIO/PREDICCION-PESOS-MAIN.git
cd PREDICCION-PESOS-MAIN

Instalar las dependencias:
pip install -r requirements.txt

Crear archivos de modelo:
python create_dummy_models.py

Nota: Si tienes los modelos entrenados reales, cópialos al directorio models/ en lugar de usar este script.



Ejecución
Para iniciar la aplicación:
python app.py
La aplicación estará disponible en http://localhost:5000
Estructura del proyecto

app.py: Aplicación principal de Flask
models/: Directorio que contiene los modelos entrenados

modelo_envios.h5: Modelo TensorFlow para predecir precios
le_inicio.pkl: LabelEncoder para ciudades de origen
le_llegada.pkl: LabelEncoder para ciudades de destino
X_train_columns.pkl: Columnas utilizadas para la codificación one-hot


create_dummy_models.py: Script para crear modelos de prueba
index.html: Interfaz de usuario
requirements.txt: Dependencias del proyecto

API
La aplicación expone un endpoint de API:

POST /predict: Recibe datos de un envío y devuelve la predicción de precio

Parámetros:

peso: Peso del paquete (número)
inicio: Ciudad de origen (string)
llegada: Ciudad de destino (string)


Ejemplo de respuesta: {"precio_predicho": "15.50 soles"}



Ciudades soportadas

Lima
Arequipa
Trujillo
Chiclayo
Piura
Cusco
Iquitos
Huancayo
Pucallpa
Tacna
Ayacucho
Chimbote
Ica
Juliaca
Tarapoto
