import cv2
import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model('mnist_digit_recognition_model')

def preprocess_image(image_path):
    img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(28,28))
    img=img/255.0
    img=img.reshape(1,28,28,1)
    return img

image_path="mnist.png"
handwritten_num=preprocess_image(image_path)

predictions=model.predict(handwritten_num)
predicted_class=np.argmax(predictions)

print("predicted digit",predicted_class)