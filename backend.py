import cv2
import numpy as np
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import base64

from testing_model import convert_image

app = Flask(__name__)
#Dashboard
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Filter Dashboard</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 40px; }
        img { max-width: 45%%; margin: 20px; border: 2px solid #ccc; }
        select, button { padding: 10px; font-size: 16px; }
    </style>
</head>
<body>
    <h1> Image Filter Dashboard </h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required><br><br>
        <select name="filter" required>
            <option value="retro">Retro</option>
            <option value="sketch">Sketch</option>
            <option value="gray">Grayscale</option>
            <option value="duotone">Duotone</option>
            <option value="thermal">Thermal</option>
            <option value="glitch">Glitch</option>
            <option value="cartoon">Cartoon</option>
            <option value="pop">Pop Art</option>
            <option value="hdr">HDR</option>
            <option value="solarize">Solarize</option>
            <option value="negative">Negative</option>
            <option value="edges">Edge Detect</option>
        </select><br><br>
        <button type="submit">Apply Filter</button>
    </form>

    {% if original_img and filtered_img %}
    <div>
        <h3>Original Image</h3>
        <img src="data:image/jpeg;base64,{{ original_img }}">
        <h3>Filtered Image ({{ filter_name }})</h3>
        <img src="data:image/jpeg;base64,{{ filtered_img }}">
    </div>
    {% endif %}
</body>
</html>
"""

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        mode = request.form['filter']
        if not file:
            return "No file uploaded", 400

        filename = secure_filename(file.filename)
        in_memory = file.read()
        img_array = np.frombuffer(in_memory, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (600, 400))

        processed = convert_image(image.copy(), mode)

        original_b64 = image_to_base64(image)
        processed_b64 = image_to_base64(processed)

        return render_template_string(HTML_PAGE, original_img=original_b64, filtered_img=processed_b64, filter_name=mode)

    return render_template_string(HTML_PAGE, original_img=None, filtered_img=None, filter_name=None)

if __name__ == '__main__':
    app.run(debug=True)  