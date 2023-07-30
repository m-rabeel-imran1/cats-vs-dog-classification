import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
train_dataMaker=ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    rotation_range=40
)
validation_dataMaker=ImageDataGenerator(rescale=1/255)
train_ds=train_dataMaker.flow_from_directory(
    r"C:\Deep Learning\cats-vs-dog classification\dogs_vs_cats\train",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)
validation_ds=validation_dataMaker.flow_from_directory(
    r"C:\Deep Learning\cats-vs-dog classification\dogs_vs_cats\test",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3),padding="valid",activation="relu",input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))

model.add(Conv2D(64,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))

model.add(Conv2D(128,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding="valid"))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1,activation="sigmoid"))
print(model.summary())
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(train_ds,epochs=2,validation_data=validation_ds)
test_img = cv2.imread("/content/cat_cat_face_cats_eyes_240527.jpg")
plt.imshow(test_img)
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
model.predict(test_input)
