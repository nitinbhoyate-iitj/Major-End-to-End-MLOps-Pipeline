from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

model, X_test, y_test = joblib.load("savedmodel.pth")

def preprocess_image(img):
    img = img.convert("L")             # grayscale
    img = img.resize((64, 64))         # Olivetti resolution
    arr = np.array(img) / 255.0
    return arr.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream)
        x = preprocess_image(img)
        prediction = int(model.predict(x)[0])

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
