import os
from flask import Flask, request, render_template, send_from_directory
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

INPUT_PATH = 'input_files'
OUTPUT_PATH = 'output_files'


@app.route('/expose/<route_key>', methods=['GET'])
def index(route_key=''):
    return render_template('serving_template.html',
                           eval_path=f'/expose/{route_key}/image')


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

    # Save Image to process
    input_filepath = os.path.abspath(os.path.join(INPUT_PATH, filename))
    output_filepath = os.path.abspath(os.path.join(OUTPUT_PATH, filename))
    file.save(input_filepath)

    img = evaluate_input(input_filepath)
    img.save(output_filepath)

    return render_template('serving_template.html',
                           eval_path=f'/expose/{route_key}/image',
                           input=f'/expose/{route_key}/input/{filename}',
                           output=f'/expose/{route_key}/output/{filename}')


@app.route('/expose/<route_key>/output/<path:path>', methods=["GET"], defaults={'directory': OUTPUT_PATH})
@app.route('/expose/<route_key>/input/<path:path>', methods=["GET"], defaults={'directory': INPUT_PATH})
def send_file(route_key='', path='', directory=''):
    return send_from_directory(directory, path)


# Load the model and run the server
if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
           "please wait until server has fully started"))

    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    load_model()

    app.run(host='0.0.0.0', threaded=False)
