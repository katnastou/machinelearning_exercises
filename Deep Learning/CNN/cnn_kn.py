# Convolutional Neural Networks (CNN)

#Installing keras
#conda install -c conda-forge keras in the anaconda prompt

#Part 1 - Building the CNN

#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = "relu" )) 
##32 feature detectors of 3x3 size
## we are using tensorflow backend - order is reversed - we first input the dimensions
## of our 2D arrays (64x64) and then the number of channels - RGB = 3

#Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#add a second layer to reduce overfitting and get better acc
classifier.add(Convolution2D(32, 3, 3, activation = "relu" )) # here we apply it on the pooled fearure maps - so no need for input shape
#Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#STep 4 -Full connection
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dense(output_dim = 1, activation = "sigmoid")) #predicted probability of 1 classcat or dog, so output_dim=1

#Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
#adapted from keras documentation impage preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #same dimensions as CNN
        batch_size=32,
        class_mode='binary') #two classes we have binary

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000, #images in the training set
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)
