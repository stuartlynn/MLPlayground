from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

image_loader = ImageDataGenerator(rescale=1./255.)
batch_size = 20

loader = image_loader.flow_from_directory("/data/tiles/", color_mode='rgb',batch_size=batch_size, target_size=(256,256), class_mode='binary')

input_image = Input(shape=(3,256,256))


x = Convolution2D(16,3,3, activation = 'relu', border_mode='same', name='conv1')(input_image)
x = MaxPooling2D( (2,2), border_mode ='same')(x)
x = Convolution2D(8,3,3, activation = 'relu', border_mode='same', name='conv2')(x)
x = MaxPooling2D((2,2),border_mode='same')(x)
x = Convolution2D(8,3,3, activation = 'relu', border_mode='same', name='conv3')(x)
encoded = MaxPooling2D((2,2),border_mode ='same')(x)


x = Convolution2D(8,3,3, activation='relu', border_mode='same', name='deconv1')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(8,3,3, activation='relu', border_mode='same', name='deconv2')(x)
x = UpSampling2D((2,2))(x)
x = Convolution2D(16,3,3, activation='relu', border_mode='same', name='deconv3')(x)
x = UpSampling2D((2,2))(x)
decoded = Convolution2D(1,3,3, activation='sigmoid', border_mode='same', name='dconv4')(x)

autoencoder = Model(input_image, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#from keras.callbacks import TensorBoard

autoencoder.fit_generator(loader,loader, nb_epoch=50)
