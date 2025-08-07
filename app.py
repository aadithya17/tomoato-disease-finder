from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = model = load_model('saved_models/tomato_disease_model.h5')


classes = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
           'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))  # adjust based on your training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return classes[class_index]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        result = predict_image(filepath)
        return render_template("index.html", prediction=result, image_path=filepath)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
