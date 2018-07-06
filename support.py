import requests
import os
import numpy as np

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers.core import RepeatVector
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import (
    Conv2D,
    UpSampling2D,
    Input,
    Reshape,
    concatenate,
)

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

import matplotlib.pyplot as plt
import tensorflow as tf


# Create embedding
def create_inception_embedding(inception, grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant', anti_aliasing=True)
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def read_img(img_id, data_dir, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(os.path.join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img


def color_result(PATH, START, END, RESULT, model, inception):
    # Make predictions on validation images
    color_me = []
    i = 0
    # Take file in range [START, END] inside the PATH folder
    for filename in os.listdir(PATH):
        if i > START and i < END:
            color_me.append(img_to_array(load_img(os.path.join(PATH, filename))))
        i += 1

    #################
    # Preprocessing #
    #################
    # From RGB to B&W and embedding
    color_me = np.array(color_me, dtype=float)
    color_me_embed = create_inception_embedding(inception, gray2rgb(rgb2gray(1.0/255*color_me)))
    color_me = rgb2lab(1.0/255*color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape+(1,))

    # Test model
    output = model.predict([color_me, color_me_embed])
    # Rescale the output from [-1,1] to [-128, 128]
    output = output * 128

    # Create the result directory if not extists
    if not os.path.exists('result'):
        os.makedirs('result')

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        # LAB representation
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        # Save images as RGB
        imsave("result/img_"+str(i)+".png", lab2rgb(cur))


def prediction_from_url(url, model, inception):
    test_image_path = '/tmp/test.jpg'

    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(test_image_path, 'wb') as f:
            f.write(response.content)

    color_me = []
    color_me.append(read_img('test', '/', 'tmp', (256, 256)))

    #################
    # Preprocessing #
    #################
    # From RGB to B&W and embedding
    color_me = np.array(color_me, dtype=float)
    color_me_embed = create_inception_embedding(inception, gray2rgb(rgb2gray(1.0/255*color_me)))
    color_me = rgb2lab(1.0/255*color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape+(1,))

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

    # B&W
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.axis('off')
    ax1.set_title('B&W')
    ax1.imshow(rgb2gray(read_img('test', '/', 'tmp', (256, 256))/255), cmap='gray')

    # Prediction
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.axis('off')
    ax2.set_title('Prediction')
    ax2.imshow(lab2rgb(cur))

    # Original
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_title('Original')
    ax3.imshow(read_img('test', '/', 'tmp', (256, 256))/255)


def load_pretrained_model(inception_wpath, colornet_wpath):
    '''Load Emil's pretrained model'''
    print('Loading pretrained model... (it could take a while)')

    # Load weights of InceptionResNet model for embedding extraction
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights(inception_wpath)
    inception.graph = tf.get_default_graph()

    # The Model
    def conv_stack(data, filters, s):
        """Utility for building conv layer"""
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        return output

    embed_input = Input(shape=(1000,))

    # Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = conv_stack(encoder_input, 64, 2)
    encoder_output = conv_stack(encoder_output, 128, 1)
    encoder_output = conv_stack(encoder_output, 128, 2)
    encoder_output = conv_stack(encoder_output, 256, 1)
    encoder_output = conv_stack(encoder_output, 256, 2)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)

    # Fusion
    # y_mid: (None, 256, 28, 28)
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output)

    # Decoder
    decoder_output = conv_stack(fusion_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = conv_stack(decoder_output, 16, 1)
    decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    # Load colornet weights
    model.load_weights(colornet_wpath)

    print('Model loaded!')
    return(model, inception)
