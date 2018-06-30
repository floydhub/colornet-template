import os
import numpy as np
from flask import Flask, send_file, request, render_template, send_from_directory
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img
from support import (
    load_pretrained_model,
    create_inception_embedding,
)

"""
Import all the dependencies you need to load the model,
preprocess your request and postprocess your result

floyd run --cpu --data floydhub/datasets/colornet/1:colornet --mode serve --env tensorflow-1.7
"""
app = Flask(__name__)
app.config['DEBUG'] = False

# MODELS
model = None
inception = None

# PATHS
INCEPTION_PATH = ('/colornet/models/'
                  'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
MODEL_PATH = '/colornet/models/color_tensorflow_real_mode_300.h5'

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

OUTPUT_PATH = 'colorized'


def load_model():
    """Load the model"""
    global model, inception
    (model, inception) = load_pretrained_model(INCEPTION_PATH, MODEL_PATH)


load_model()


def allowed_file(filename):
    """Sanity check on file extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def data_preprocessing(input_filepath):
    """From RGB image to L(grayscale)"""
    global inception

    img = image.load_img(input_filepath, target_size=(256, 256))
    img = image.img_to_array(img)
    color_me = [img]

    #################
    # Preprocessing #
    #################
    # From RGB to B&W and embedding
    color_me = np.array(color_me, dtype=float)
    color_me_embed = create_inception_embedding(inception, gray2rgb(rgb2gray(1.0/255*color_me)))
    color_me = rgb2lab(1.0/255*color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape+(1,))
    return (color_me, color_me_embed)


@app.route('/expose/<route_key>/hello', methods=['GET'])
def index(route_key):
    return render_template('serving_template.html')


@app.route('/expose/<route_key>/hello', methods=["POST"])
def evaluate(route_key):
    """"Preprocessing the data and evaluate the model"""
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']

    if file.filename == '':
        return BadRequest("File name is not present in request")

    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")

    filename = secure_filename(file.filename)

    # Save Image to process
    input_filepath = os.path.abspath(os.path.join('tmp', filename))
    output_filepath = os.path.abspath(os.path.join(OUTPUT_PATH, filename))
    file.save(input_filepath)

    # # Test with noop response
    # return render_template('serving_template.html', colorized=f'/img/{filename}')

    (color_me, color_me_embed) = data_preprocessing(input_filepath)

    global model

    # Test model
    output = model.predict([color_me, color_me_embed])
    # Rescale the output from [-1,1] to [-128, 128]
    output = output * 128

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        # LAB representation
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]

    # Save image
    img = array_to_img(lab2rgb(cur))
    img.save(output_filepath)

    return render_template('serving_template.html', colorized=f'/expose/{route_key}/colorized/{filename}')
    # Send output to Client
    # return send_file(os.path.join(OUTPUT_PATH, filename),
    #                  mimetype='image/png')


@app.route('/expose/<route_key>/colorized/<path:path>', methods=["GET"])
def send_js(route_key='', path=''):
    # return send_from_directory('tmp', path)
    return send_from_directory(OUTPUT_PATH, path)


# Load the model and run the server
if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
           "please wait until server has fully started"))
    app.run(host='0.0.0.0', threaded=False)
