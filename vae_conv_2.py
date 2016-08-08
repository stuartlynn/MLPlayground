'''This script demonstrates how to build a variational autoencoder with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
import keras

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.preprocessing.image import ImageDataGenerator
import glob
from PIL import Image
# input image dimensions
img_rows, img_cols, img_chns = 256, 256, 3
# number of convolutional filters to us
nb_filters = 16
# convolution kernel size
nb_conv = 3

batch_size = 50
original_dim = (img_chns, img_rows, img_cols)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 50

# print 'loading data'
# def load_data():
#   files = glob.glob('tiles/train/*.png')
#   batch = []
#   for filename in files:
#      img = Image.open(filename)
#      norm = np.asarray(img, dtype=np.float)/255
#      batch.append(norm)
#   return batch 


#data = load_data()


x = Input(batch_shape=(batch_size,) + original_dim)
c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu')(x)
f = Flatten()(c)
h = Dense(intermediate_dim, activation='relu')(f)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_f = Dense(nb_filters*img_rows*img_cols, activation='relu')
decoder_c = Reshape((nb_filters, img_rows, img_cols))
decoder_mean = Deconvolution2D(img_chns, nb_conv, nb_conv,
                               (batch_size, img_chns, img_rows, img_cols),
                               border_mode='same')

h_decoded = decoder_h(z)
f_decoded = decoder_f(h_decoded)
c_decoded = decoder_c(f_decoded)
x_decoded_mean = decoder_mean(c_decoded)

def autoGenerator():
  image_loader = ImageDataGenerator(rescale=1./255.)
  loader = image_loader.flow_from_directory("/data/tiles/", color_mode='rgb',batch_size=batch_size, target_size=(256,256), class_mode='binary')
  for batch in loader:
    if np.isnan(batch[0]).any():
	print 'problem with batch'
    yield (batch[0],np.copy(batch[0])) 


# print 'here'
# next(
# for batch in loader:
# 	print np.array(batch).shape
# print 'here'


def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = np.dot(original_dim, original_dim) * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()

# for batch in autoGenerator():
#    if np.isnan(batch[0]).any():
#      print 'we have an issue'

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train.astype('float32')[:, None, :, :] / 255.
#x_test = x_test.astype('float32')[:, None, :, :] / 255.

vae.fit_generator(autoGenerator(), batch_size  ,
        nb_epoch=nb_epoch)	


with open('model.json','w') as model_file:
  model_file.write(vae.to_json())

vae.save_weights("model.h5")

import sys
sys.exit()

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_f_decoded = decoder_f(_h_decoded)
_c_decoded = decoder_c(_f_decoded)
_x_decoded_mean = decoder_mean(_c_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
#n = 15  # figure with 15x15 digits
#digit_size = 28
#figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
#grid_x = np.linspace(-15, 15, n)
#grid_y = np.linspace(-15, 15, n)

#for i, yi in enumerate(grid_x):
#    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        x_decoded = generator.predict(z_sample)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#       figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit

#plt.figure(figsize=(10, 10))
#plt.imshow(figure)
#plt.show()
