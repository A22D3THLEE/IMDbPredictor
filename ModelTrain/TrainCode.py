import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.data_utils import pad_sequences
from keras_preprocessing.text import Tokenizer

num_words = 10000
max_review_len = 200

train = pd.read_csv('train.csv',
                    header=None,
                    names=['Class', 'Review'])
reviews = train['Review']
y_train = train['Class'] - 1
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)

x_train = pad_sequences(sequences, maxlen=max_review_len)

model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_review_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model_save_path = 'best_model.h5'
checkpoint_callback = ModelCheckpoint(model_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)
history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[checkpoint_callback])
