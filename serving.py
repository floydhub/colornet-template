import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from support import (
    load_pretrained_model,
    create_inception_embedding,
)


INCEPTION_PATH = ('/colornet/models/'
                  'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
MODEL_PATH = '/colornet/models/color_tensorflow_real_mode_300.h5'

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])


# MODELS
model = None
inception = None


def load_model():
    """Load the model"""
    global model, inception
    (model, inception) = load_pretrained_model(INCEPTION_PATH, MODEL_PATH)


def evaluate_input(input: str):
    global model
    (color_me, color_me_embed) = _data_preprocessing(input)
    output = model.predict([color_me, color_me_embed])
    # Rescale the output from [-1,1] to [-128, 128]
    output = output * 128
    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        # LAB representation
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
    img = array_to_img(lab2rgb(cur))
    return img


def _data_preprocessing(input_filepath):
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
