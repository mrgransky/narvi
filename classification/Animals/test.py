import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    

model = tf.keras.models.load_model("model_0_64_3.model")

"""
prediction = model.predict([prepare('/home/xenial/WS_Farid/keras_ws/classification/cat_or_dog/query_images/Dog/4.jpg')])
"""
prediction = model.predict([prepare('/home/xenial/WS_Farid/keras_ws/classification/cat_or_dog/query_images/Dog/2.jpg')])



print "\n\nPrediction: ", CATEGORIES[int(prediction[0][0])]
print "\n\nDONE!"


