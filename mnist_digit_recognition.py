import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Step 1 : I have loaded the data
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

# Step 2: preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshaping image
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model=models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10),
    layers.BatchNormalization(),
    layers.Activation('softmax'),
])

# Step 4: Compile the Model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

# Step 5: Train Model
model.fit(x_train,y_train,epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Step 6: Evaluate Model
test_loss,test_accuracy=model.evaluate(x_test,y_test)
print(f"Test Accuracy:{test_accuracy}")

# Step 7: Save the trained model
model.save("mnist_digit_recognition_model_Batch_normal")

# Step 8: Load the saved model
loaded_model = tf.keras.models.load_model("mnist_digit_recognition_model")

# Step 9: Make predictions using the loaded model
predictions = loaded_model.predict(x_test)

## without batch normalization accuracy is 0.99029
## with batch normalization accuracy is 0.99279