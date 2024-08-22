from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io

# App Instance
app = Flask(__name__)
CORS(app)

@app.route("/api/upload", methods=['POST'])
def upload_image():
	if 'image' not in request.files:
		return jsonify({"message": "No image part in the request"}), 400

	image_file = request.files['image']
	if image_file.filename == '':
		return jsonify({"message": "No selected file"}), 400

	try:
		# Open and resize the image
		img = Image.open(image_file.stream)
		img = img.resize((64, 64))

		# Load the model and make a prediction
		model = tf.keras.models.load_model('Model/cnn_model.keras')
		test_image = keras_image.img_to_array(img)
		test_image = np.expand_dims(test_image, axis=0)
		result = model.predict(test_image)

		print(result)

		prediction = 'This is a dog.' if result[0][0] == 1 else 'This is a cat.'

		return jsonify({"message": "Image uploaded and Detected successfully", "prediction": prediction}), 200

	except Exception as e:
		return jsonify({"message": f"Failed to process image: {str(e)}"}), 500
			

if __name__ == '__main__':
    app.run(debug=True, port=3500)
