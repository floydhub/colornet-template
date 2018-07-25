import os
import base64
from io import BytesIO

from flask import Flask, make_response, request, render_template
from werkzeug.exceptions import BadRequest
from serving import (
    load_model,
    evaluate_input,
)

"""
floyd run --cpu --data floydhub/datasets/colornet/1:colornet --mode serve --env tensorflow-1.7
"""
app = Flask(__name__)
app.config['DEBUG'] = False
load_model()


@app.route('/', methods=['GET'])
def index():
    return render_template('serving_template.html')


@app.route('/image', methods=["POST"])
def eval_image():
    """"Preprocessing the data and evaluate the model"""
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")
    if input_file.filename == '':
        return BadRequest("File name is not present in request")
    if not input_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")

    # # Save Image to process
    input_buffer = BytesIO()
    output_buffer = BytesIO()
    input_file.save(input_buffer)

    img = evaluate_input(input_buffer)
    img.save(output_buffer, format="JPEG")
    img_str = base64.b64encode(output_buffer.getvalue())

    response = make_response(img_str)
    response.headers.set('Content-Type', 'image/jpeg')
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=False)
