#Using Bert to detect sentiment in text

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

dataset, info = tfds.load('sentiment140', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec

for example, label in train_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_data = train_dataset.take(4000)

# print(type(train_dataset))
# print(len(train_data))
# print(len(test_dataset))

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())

embedder = tf.keras.layers.Embedding(input_dim = 1000, output_dim = 64, mask_zero = True) #converts series of words into series of vectors depending on encoder
bidirectional_layer_64 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)) #front and back propagation
bidirectional_layer_32 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
dense_layer_1 = tf.keras.layers.Dense(64, activation = 'relu')
dense_layer_2 = tf.keras.layers.Dense(1)

model = tf.keras.Sequential()#stackes layers on top of each other
model.add(encoder)
model.add(embedder)
model.add(bidirectional_layer_64)
model.add(bidirectional_layer_32)
model.add(dense_layer_1)
model.add(tf.keras.layers.Dropout(0.5))
model.add(dense_layer_2)

#FLAG: try and learn about the stateful RNN layer to see which one is better

#Compiling model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])

#Train model
history = model.fit(train_data, epochs=10, validation_data=test_dataset, validation_steps=30)

#Testing
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

