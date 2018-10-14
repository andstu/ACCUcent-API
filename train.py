import numpy as np
from tensorflow import keras
import tensorflow as tf

train = np.load('train.npy')
trainLabel = np.load('trainLabel.npy')
test = np.load('test.npy')
testLabel = np.load('testLabel.npy')

print("A")
model = keras.Sequential({
    keras.layers.Dense(128, activation='tanh', input_shape=(len(train) + len(test), )),
    keras.layers.Dense(53, activation=tf.nn.softmax) #Hard
})
print("B")
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("C")
model.fit(train, trainLabel, epochs=5, batch_size=5)
print("D")
test_loss, test_acc = model.evaluate(test, testLabel)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)