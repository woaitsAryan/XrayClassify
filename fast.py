import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = tf.keras.models.load_model('model/vgg16_best.h5')

img_path = 'data/test/Covid/0112.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)

max_index = preds.argmax()

print(preds)
if max_index == 0:
    print("Normal")
elif max_index == 1:
    print("Viral Pneumonia")
elif max_index == 2:
    print("Covid")

