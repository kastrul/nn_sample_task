from keras import Sequential
from keras.constraints import nonneg
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, GaussianNoise
from keras.models import model_from_json


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(r'../models/' + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(r'../models/' + name + ".h5")
    print("Saved model to disk")


def load_model(name):
    # load json and create model
    json_file = open(r'../models/' + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r'../models/' + name + ".h5")
    print("Loaded model from disk")

    return loaded_model


def lstm_sequence_model(x):
    # design network
    model = Sequential()

    model.add(LSTM(150,
                   input_shape=(x.shape[1], x.shape[2]),
                   return_sequences=True,
                   ))
    model.add(GaussianNoise(2.0))
    model.add(LSTM(75, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu'))
    model.add(Dropout(0.25))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


def lstm_simple_model(train_x):
    # design network
    model = Sequential()
    model.add(LSTM(75,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True
                   ))
    model.add(LSTM(150, return_sequences=False))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_constraint=nonneg()))
    model.compile(loss='mae', optimizer='adam')
    return model


def stateful_lstm_model(train_x, batch_size, stateful: bool):
    model = Sequential()
    model.add(LSTM(75,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   batch_size=batch_size,
                   stateful=stateful))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, kernel_constraint=nonneg()))
    model.compile(loss='mse', optimizer='adam')
    return model


def regular_model(train_x):
    model = Sequential()
    model.add(Dense(150,
                    activation='relu',
                    input_shape=(train_x.shape[1], train_x.shape[2])
                    ))
    model.add(Dropout(0.2))
    model.add(Dense(225, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model


def conv_model(train_x):
    model = Sequential()
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     input_shape=(train_x.shape[1], train_x.shape[2]),
                     activation='relu'
                     ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    # model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model
