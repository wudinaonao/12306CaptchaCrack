from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output model")
# ap.add_argument("-l", "--labelbin", required=True,
#                 help="path to output label binarizer")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())

model_path = os.path.join(os.getcwd(), "model", "12306.model")
labelbin_path = os.path.join(os.getcwd(), "label", "12306.label")
# image_path = os.path.join(os.getcwd(), "c78e69057e9441d036a7cac077402d76.png")
image_path_list = []
for root, dirs, imgs in os.walk(os.path.join(os.getcwd(), "test_data")):
    for img_path in imgs:
        image_path_list.append(os.path.join(root, img_path))
        
print("[INFO] loading network...")
model = load_model(model_path)
lb = pickle.loads(open(labelbin_path, 'rb').read())

print("[INFO] classfifying image...")
error_count = 0
for image_path in image_path_list:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),cv2.IMREAD_COLOR)
    # output = image.copy()
    
    image = cv2.resize(image, (67, 67))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    
    image_name = os.path.basename(image_path).split("_")[0].strip()
    if label.strip() != image_name:
        error_count += 1
        print("Original: {}    Prediction:{}".format(image_name, label))
    label = "{}: {:.2f}%".format(label, proba[idx]*100)
    # output = imutils.resize(output, width=400)
    # cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (0, 255, 0), 2)
    print("[INFO] {}".format(label))
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
print("Test Accuracy:", str(round((len(image_path_list) - error_count)/len(image_path_list), 2)))