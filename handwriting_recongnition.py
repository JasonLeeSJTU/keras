from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# import data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = np.array(train_x, dtype=float)/255.0
print(train_x.shape)
train_x = train_x.reshape((60000, 28, 28, 1))

test_x = np.array(test_x, dtype=float)/255.0
print(test_x.shape)
test_x = test_x.reshape(10000, 28, 28, 1)

print(train_y.shape)
print(test_y.shape)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# validation set
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y,
                                                  test_size=0.4)

# CNN
model = Sequential()
# convolution layers
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
# pooling layers
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# tensors to vectors
model.add(Flatten())
model.add(Dense(256))

# output layer
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer=RMSprop(lr=0.001), loss="categorical_crossentropy",
        metrics=['accuracy'])

# start training
H = model.fit(x_train, y_train, epochs=10, batch_size=128,
        validation_data=(x_val, y_val))

# plot
plt.style.use("ggplot")
plt.figure()
N=10
plt.plot(np.arange(0,N), H.history["loss"], linestyle='--', color='r',
        label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], color='b', label="val_loss")
plt.title("loss")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Loss.png")

plt.figure()
N=10
plt.plot(np.arange(0,N), H.history["acc"], linestyle='--', color='r',
        label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], color='b', label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Accuracy.png")

# estimation
test_loss, test_acc = model.evaluate(test_x, test_y)
print("test_loss: ", test_loss)
print("test_acc: ", test_acc)
