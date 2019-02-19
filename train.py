import matplotlib
matplotlib.use("Agg")

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import time

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
#                 help="path to output label binarizer")
# ap.add_argument("-p", "--plot", required=True,
#                 help="path to output accuracy/loss plot")
# args = vars(ap.parse_args())

DATASET_PATH = os.path.join(os.getcwd(), "data", "dataset")
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model", "12306_eachDir3000.model")
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "model", "save_cache")
LABELBIN_SAVE = os.path.join(os.getcwd(), "label", "12306.label")
LOSS_PLOT_PATH = "accuracy_and_loss"

EPOCHS = 1000
INIT_LR = 1e-3
BS = 8
IMAGE_DIMS = (67, 67, 3)

data = []
labels = []

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8),cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)
))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.2,
                                                  random_state=42)
aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1],
                            height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2],
                            classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("[INFO] training network...")

model_cache_path = os.path.join(MODEL_CACHE_DIR,"model_cache.model")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[
        ModelCheckpoint(
            model_cache_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1),
        # EarlyStopping(patience=50),
        ReduceLROnPlateau(patience=10),
        CSVLogger("training.log")
        ]
    )

print("[INFO] serializing network...")
model.save(MODEL_SAVE_PATH)

print("[INFO] serializing label binarizer...")
with open(LABELBIN_SAVE, "wb") as f:
    f.write(pickle.dumps(lb))

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(LOSS_PLOT_PATH)
