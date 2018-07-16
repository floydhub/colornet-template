import os
import base64
from io import BytesIO

from flask import Flask, send_file, make_response, request, render_template, send_from_directory
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from serving import (
    load_model,
    evaluate_input,
)

"""
floyd run --cpu --data floydhub/datasets/colornet/1:colornet --mode serve --env tensorflow-1.7
"""
app = Flask(__name__)
app.config['DEBUG'] = False

# load_model()

@app.route('/', methods=['GET'])
@app.route('/expose/<route_key>', methods=['GET'])
def index(route_key=''):
    path = '/image' if route_key == '' else f'/expose/{route_key}/image'
    return render_template('serving_template.html', eval_path=path)


@app.route('/image', methods=["POST"])
@app.route('/expose/<route_key>/image', methods=["POST"])
def eval_image(route_key=''):
    """"Preprocessing the data and evaluate the model"""
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return BadRequest("Invalid file type")
    filename = secure_filename(file.filename)

    # # Save Image to process
    input_buffer = BytesIO()
    output_buffer = BytesIO()
    file.save(input_buffer)

    img = evaluate_input(input_buffer)
    img.save(output_buffer, format="JPEG")
    img_str = base64.b64encode(output_buffer.getvalue())

    response = make_response(img_str)
    response.headers.set('Content-Type', 'image/jpeg')
    return response


if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
           "please wait until server has fully started"))

    load_model()

    app.run(host='0.0.0.0', threaded=False)
