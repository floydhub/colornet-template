import os
import numpy as np

import flask
from flask import Flask
from flask import Flask, send_file, request

from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from support import load_pretrained_model, create_inception_embedding, read_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb

from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

"""
Import all the dependencies you need to load the model,
preprocess your request and postprocess your result
"""
app = Flask(__name__)
app.config['DEBUG'] = False

# MODELS
model = None
inception = None

# PATHS
INCEPTION_PATH = '/colornet/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
MODEL_PATH = '/colornet/models/color_tensorflow_real_mode_300.h5'

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

# EVAL_PATH = '/tmp'
OUTPUT_PATH = '/output'


def load_model():
	"""Load the model"""
	global model, inception
	(model, inception) = load_pretrained_model(INCEPTION_PATH, MODEL_PATH)

def allowed_file(filename):
	"""Sanity check on file extension"""
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def data_preprocessing(input_filepath):
	"""From RGB image to L(grayscale)"""
	global inception

	color_me = []
	color_me.append(read_img('test', '/', 'output', (256, 256)))

	#################
	# Preprocessing #
	#################
	# From RGB to B&W and embedding
	color_me = np.array(color_me, dtype=float)
	color_me_embed = create_inception_embedding(inception, gray2rgb(rgb2gray(1.0/255*color_me)))
	color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
	color_me = color_me.reshape(color_me.shape+(1,))

	return (color_me, color_me_embed)

# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send
# arbitrary data to the endpoint in the request body,
# including images, JSON, encoded-data, etc.)
@app.route('/<path:path>', methods=["POST"])
def evaluate(path):
	""""Preprocessing the data and evaluate the model"""
	if flask.request.method == "POST":
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
		input_filepath = os.path.join(OUTPUT_PATH, 'test.jpg')
		file.save(input_filepath)

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
			cur[:,:,0] = color_me[i][:,:,0]
			cur[:,:,1:] = output[i]

		# Save image
		img = array_to_img(lab2rgb(cur))
		img.save(os.path.join(OUTPUT_PATH, 'output.png'))

		# Send output to Client
		return send_file(os.path.join(OUTPUT_PATH, 'output.png'), mimetype='image/png')

# Load the model and run the server
if __name__ == "__main__":
	print(("* Loading model and starting Flask server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0', threaded=False)
