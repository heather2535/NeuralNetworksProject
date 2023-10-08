import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import imghdr
from matplotlib import pyplot as plt

data_dir = 'data'

image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#   1. Remove dodgy images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Load Data
import numpy as np
from matplotlib import pyplot as plt
data = tf.keras.utils.image_dataset_from_directory('data')  # building our data pipeline
data_iterator = data.as_numpy_iterator()
# Get another batch from the iterator
batch = data_iterator.next()
# Images represented as numpy arrays
print(batch[0].shape)

#   Class 1 = SAD PPL
#   Class 0 = HAPPY PPL
print(batch[1]) # 0 and 1 represents either happy or sad

#   check which class is assigned to which type of image
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
#plt.show()


#   2. Preprocess Data
scaled = (batch[0] / 255)
print(scaled.min())
print(scaled.max())

#   2.1 Scale Data
data = data.map(lambda x,y: (x/255, y))   # data.map allows transformation to be formed in pipeline
                                          # x = image, y= labels; no transformation on y
scaled_iterator = data.as_numpy_iterator()
batch= scaled_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
#plt.show()

#   2.2 Split Data:
# when validating data, and ensuring model will not be over fit, we have specific partitions.
'''
    Over fitting occurs when the model cannot generalize and fits too closely to the training dataset instead. 
    Over fitting happens due to several reasons, such as:
•    The training data size is too small and does not contain enough data samples to accurately represent all possible input data values.
•    The training data contains large amounts of irrelevant information, called noisy data.
•    The model trains for too long on a single sample set of data.
•    The model complexity is high, so it learns the noise within the training data.

Overfitting examples
Consider a use case where a machine learning model has to analyze photos and identify the ones 
that contain dogs in them. If the machine learning model was trained on a data set that contained 
majority photos showing dogs outside in parks , it may may learn to use grass as a feature for 
classification, and may not recognize a dog inside a room.
'''
print(len(data))
train_size = int(len(data)*.7)  # training set is 70% of data
val_size = int(len(data)*.2)    # validation is 20% of data;
test_size = int(len(data)*.1)   # test size is 10% of data
'''
training data is what is used to train our deep learning model
validation data is going to be used to evaluate model while we are training. 
test partition is not going to have seen until we get to final evaluation stage. until end. 
train_size and val_size both used DURING TRAINING, test_size used POST TRAINING for evaluation
'''
print(train_size)
print(val_size)
print(test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#   Deep Model
#   3.1 Build Deep Learning Model
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

#   Conv2D is a 2D convolution layer(spatial convolution over images)
#   MaxPooling2D acts as a condensing layer
#   Dense: The Dense layer in Keras is a good old, fully/densely-connected neural network.
#   Flatten: flattens any input into 1D vector
#   Dropout: a regularization technique that randomly sets a fraction of the input units
#           to 0 at each training update, effectively dropping out a portion of the neurons
#           during training. By doing so, dropout reduces the interdependency among neurons
#           This helps prevent overfitting by randomly selecting some neurons to 'turn off'.

model = Sequential() # initialize model
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu',))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu',))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
print(model.summary())

#   3.2 Train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#   3.3 Plot Performance
# plots loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# plots accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='loss')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# 4. Evaluation
#   4.1 Evaluate:
import keras
import tensorflow as tf
from keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
'''
Precision shows how often an ML model is correct when predicting the target class.
Recall shows whether an ML model can find all objects of the target class. 
Accuracy shows how often a classification ML model is correct overall. 

'''

for batch in test.as_numpy_iterator:
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, BinaryAccuracy:{acc.result().numpy()}')


# 4.2 Test
# read in an image that the model has never seen before
#happy test
img = cv2.imread('happytest.jpeg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

np.expand_dims(resize, 0).shape
yhut = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


#   Save the model
import keras
import tensorflow as tf
from keras.models import load_model
model.save(os.path.join('models', 'happysadmode1.h5'))
new_model = load_model(os.path.join('models', 'happysadmode1.h5'))
new_model