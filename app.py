from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the CNN model
model = tf.keras.models.load_model("face_mask.h5")


# Preprocess input image
def preprocess_image(image):
    img = Image.open(image).convert("RGB")  # Ensure 3 color channels
    img = img.resize((128, 128))           # Resize to model's input size
    img_array = np.array(img) / 255.0      # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route("/")
def index():
    return render_template("index.html")  # Renders the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]

    try:
        # Preprocess the image and make a prediction
        preprocessed_image = preprocess_image(file)
        predictions = model.predict(preprocessed_image)

        # Binary classification logic
        confidence = predictions[0][0]
        predicted_class = 1 if confidence >= 0.5 else 0

        # Map the class index to text
        if predicted_class == 1:
            prediction_text = "Not wearing a mask"
        else:
            prediction_text = "Wearing a mask"

        return jsonify({
            "prediction": prediction_text,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
