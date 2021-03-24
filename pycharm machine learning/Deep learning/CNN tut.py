'''
#####
Data preprocessing
####
'''

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras_preprocessing import image

'''
preprocessing the Training set
'''

# keras API
train_datagen = ImageDataGenerator(
        rescale=1./255, #feature scalling to each pixel
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)# the last 3 arguments are the transforamtion that we will perform to prevent overfitting
train_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32, # how many images in each batch
        class_mode='binary')# binay or categorical

'''
preprocessing the Test set
'''

test_datagen = ImageDataGenerator(rescale=1./255) # we just need to scale them
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

'''
####
building the CNN
####
'''

cnn = ann = tf.keras.models.Sequential()

'''
convolution
'''

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

'''
pooling(max)
'''

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

'''
adding a second convolutional layer
'''

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

'''
flattening
'''

cnn.add(tf.keras.layers.Flatten())

'''
Full connection
'''

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

'''
output layer
'''

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))# binary classification

'''
####
Training the CNN
####
'''

'''
compiling the CNN
'''

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''
training the CNN on the training set and evaluating it on the test set
'''

cnn.fit(x=train_set, validation_data=test_set, epochs=25)# you need to try the epochs before choosing the best one

'''
####
making a single pred
####
'''

test_image = image.load_img(path='single_prediction//cat_or_dog_1.jpg', target_size=(64, 64))#we need to resize this image to the same as in the training
# convert this image to an numpy array

test_image = image.img_to_array(test_image)

#this image need to be in a batch(add an extra Dimension)

test_image = np.expand_dims(test_image, axis=0)# the D of the batch is the first Dimension
result = cnn.predict(test_image)
train_set.class_indices #dogs1 ,cats 0
if result[0][0]==1:#acc the batch and the single element in this batch
        prediction = 'Dog'
else:
        prediction= 'Cat'
print(prediction)




