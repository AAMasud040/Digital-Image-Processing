import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve,roc_auc_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
details = tf.config.experimental.get_device_details(physical_devices[0])
print(details.get('device_name', 'Unknown GPU'))

tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG19_MODEL=tf.keras.applications.VGG19(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')


for l in VGG19_MODEL.layers:
  l.trainable = False


x = layers.Flatten()(VGG19_MODEL.output)
dl1 = layers.Dense(512,activation='relu')(x)
dl2 = layers.Dense(256,activation='relu')(dl1)
dl3 = layers.Dense(128,activation='relu')(dl2)
dl4 = layers.Dense(64,activation='relu')(dl3)
pred = layers.Dense(6,activation='sigmoid')(dl4)



Fmodel = Model(inputs=VGG19_MODEL.input, outputs = pred)
Fmodel.summary()


Fmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
              loss = tf.keras.losses.categorical_crossentropy,
              metrics= ['categorical_accuracy'])

print("safe model 1")
#-------------------------------------------------------------------------------
img_dir = ['./resized_256_256_3/colon_aca/colonca',
            './resized_256_256_3/colon_n/colonn',
            './resized_256_256_3/lung_aca/lungaca',
            './resized_256_256_3/lung_n/lungn',
            './resized_256_256_3/lung_scc/lungscc']
label = [1,2,3,4,5]

dataset = []

for folder in img_dir:
  label_class = label[img_dir.index(folder)]
  print(label_class)
  for i in range(1,25000):
    photo =  folder+str(i)+".jpeg"
    image = cv2.imread(photo,cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_NEAREST)
    image = image.astype('float32') / 255.
    dataset.append([np.array(image),np.array(label_class)])


print('done')

x = np.array([i[0] for i in dataset]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in dataset])
print(x.shape)
print(y.shape)

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.15)
print(x_val.shape)
print(y_val.shape)


y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)

print(y_train_encoded[0])
print("safe model 2")
# y_train_encoded[0]
history = Fmodel.fit(x_train, y_train_encoded, validation_data =(x_val, y_val_encoded), epochs=3, batch_size=1)
print("safe model 3")

def plotHistory(history):
  pd.DataFrame(history.history).plot(figsize=(8,8))
  plt.grid(True)
  plt.gca().set_ylim(0,1.0)
  plt.show()

plotHistory(history)
print("safe model 4")
y_pred_train = np.argmax(Fmodel.predict(x_train), axis=-1)
y_pred_val = np.argmax(Fmodel.predict(x_val), axis=-1)

fpr , tpr , thresholds = roc_curve ( y_val , y_pred_val)

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
# plot_roc_curve (fpr,tpr)

# auc_score=roc_auc_score(y_val,y_pred_val)
# print(auc_score)
# print("safe model 5")
# print(classification_report(y_train, y_pred_train))
# print(classification_report(y_val, y_pred_val))