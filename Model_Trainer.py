# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v2 as tf

# Helper libraries
import numpy as np
import os
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

def percentSlice(array, percentStart, PercentEnd):
    """Returns a percentage of an array"""
    return array[int(len(array) * percentStart) : int(len(array) * PercentEnd)]



#pull folder for training data
data = os.listdir('data')
#Init array varibles to store formated data
randomized = []

#Make sure the folder can open all the files before formating data
for folder in data:
    print(folder)
    files = os.listdir('data/'+folder)
    print("files are good")

#pulls image files from libaray  
for folder in data:
    files = os.listdir('data/'+folder)
    classRN = 1 if "healthy"in folder else 0
    i = 0
    z = 0
    print(folder)
    lastValue = classRN
    print(classRN, z)
    if False:
        pass 
    else:
        for f in files:
            #For the diseased2 folder, pull every other file
            if not("healthy"in folder) and "2" in folder:
                i += 1
                if z > 70000: #57000
                    break
                if i > 1:
                    i = 0
                else:
                    continue

            #open and format image
            img = Image.open( "data/"+folder+"/"+f )#.convert('LA')
            img = img.resize((80,80))
            new_array = np.array(img)

            #append class (healthy v unhealthy) and numpy array of the image
            randomized.append([1 if "healthy"in folder else 0, [new_array]])

            #For the smaller folders, go to the next folder at 2200 images
            if not("2" in folder):
                if z > 2200: #1900
                    break
            #for healthy2, return at 18000 iamges
            elif z > 18000: #18000
                break
            z += 1   
print("done")

#Randomize dataset to ensure order of files does not impact neural's nets decsions
randomized = np.asarray(randomized)
np.random.shuffle(randomized)

#Spilt arrays into thier labels and data to work with Tensorflow
classes = []
plants = []
for _ in range(len(randomized)):
    try:
        plants.append(tf.constant(randomized[_][1][0], shape = (80, 80, 3)))
        classes.append(tf.constant(randomized[_][0], shape = (1)))
    except:
        print("issue with",randomized[_][1][0])
        #NOTE: There is an issue with a few images in the data set in which the numpy array isn't returned as (rows, col, channels) format. 
        #the soultion is to skip these iamges when building the dataset
        continue
    
print("number of classes", len(classes), classes[0])
print("number of vectors:", len(plants), plants[0])

#Devide the list of plant data and classes into training and testing datasets
train_examples = percentSlice(plants, 0, 0.75)
train_labels = percentSlice(classes, 0, 0.75)
print(len(train_labels), len(train_examples))

test_examples = percentSlice(plants, 0.75, 1)
test_labels = percentSlice(classes, 0.75, 1)
print(len(test_labels), len(test_examples))

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
print(test_dataset)


#https://www.tensorflow.org/tutorials/images/classification#predict_on_new_data
#The code from here uses tensorflow tutorials as a referance for image classiflication 

#Devide the set into batches to check each epoch
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print("starting layers")
layers = tf.keras.layers

#randomize orentiation to avoid overfitting
data_aug = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(80,80, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
])

#process images in neural net to find patterns in the RGB values
model = tf.keras.Sequential([
  data_aug,
  layers.experimental.preprocessing.Rescaling(1./255), 
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'), #64
  layers.Dense(2) #reutrns one of two classes, healthy or unhealthy (1 or 0)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Test and train the model
numOfepochs = 15

history = model.fit(train_dataset, validation_data=test_dataset, epochs=numOfepochs) #175 #50
model.evaluate(test_dataset)

#save model to use on plant checkup website
if (input("save? y/n   ") == "y"):
    model.save('leaf_health_classififer') 

#Make maitlab graph for presentation 
if (input("graph training process?  y/n:  ") == "y"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs_range = range(numOfepochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Testing Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy over time')
    plt.show()