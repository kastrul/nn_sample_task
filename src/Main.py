from numpy.ma import sqrt
from sklearn.metrics import mean_squared_error

from DataReader import get_scaled_arrays, get_unit_batches, get_data_frames, get_data_by_id, get_unit_by_id
from ModelCreation import *
from Predictor import *
from Training import *

train_df, test_df = get_data_frames()

train_x, train_y, test_x, test_y = get_scaled_arrays(train_df, test_df)
ignored_columns = ['sensor ' + str(i) for i in [1, 5, 6, 10, 7, 9, 12, 14, 15, 16, 18, 19, 20, 21]]
ignored_columns.append('setting 1')
ignored_columns.append('setting 2')
ignored_columns.append('setting 3')
train_xid, train_yid, test_xid, test_yid = get_data_by_id('FD001', train_df, test_df, ignore=ignored_columns)
train_xuid, train_yuid, test_xuid, test_yuid = get_unit_by_id('FD001', 1, train_df, test_df, ignore=ignored_columns)
train_bx, train_by, test_bx, test_by = get_unit_batches(train_df, test_df)


def get_shaped_array(array_x, array_y, window):
    length = array_x.shape[0]
    modulus = length % window
    if modulus != 0:
        array_x = array_x[:- modulus]
        array_y = array_y[:- modulus]
    new_length = int((length - modulus) / window)
    array_x = array_x.reshape(new_length, window, array_x.shape[2])
    array_y = array_y[::window]
    return array_x, array_y


def predicting(model_name, test_batch_x, test_batch_y):
    # predicting from model
    y_predict = lstm_predictor(test_batch_x, model_name)
    rmse = sqrt(mean_squared_error(test_batch_y.reshape(test_batch_y.shape[0], 1),
                                   y_predict.reshape(y_predict.shape[0], 1)
                                   ))
    print('Test RMSE: {}'.format(rmse))


def training(model_name, trainer, epochs=500, batch_size=100):
    # model training
    if trainer == 'lstm':
        x_train = train_x
        x_test = test_x
        y_train = train_y
        y_test = test_y
        model = lstm_simple_model(x_train)
        lstm_trainer(x_train, y_train, x_test, y_test,
                     model, model_name, epochs, batch_size)
    elif trainer == 'conv':
        x_train = train_x[:-9]
        x_test = test_x[:-7]

        y_train = train_y[:-9:10]
        y_test = test_y[:-7:10]

        x_train = x_train.reshape(int(x_train.shape[0] / 10), 10, 25)
        x_test = x_test.reshape(int(x_test.shape[0] / 10), 10, 25)

        y_train = y_train.reshape(y_train.shape[0], 1, 1)
        y_test = y_test.reshape(y_test.shape[0], 1, 1)
        model = conv_model(x_train)
        conv_trainer(x_train, y_train,
                     x_test, y_test,
                     model, model_name,
                     epochs, batch_size)

    elif trainer == 'stateful_lstm':
        stateful = True
        model = stateful_lstm_model(train_x, batch_size, stateful)
        stateful_lstm_trainer(train_x, train_y,
                              test_x, test_y,
                              model, model_name,
                              epochs, batch_size)
    elif trainer == 'lstm_by_id':
        x, y = get_shaped_array(train_xid, train_yid, 5)
        x_test, y_test = get_shaped_array(test_xid, test_yid, 5)
        model = lstm_simple_model(x)
        lstm_trainer(x, y, x_test, y_test,
                     model, model_name, epochs, batch_size)


def main(c):
    if c == 1:
        training('model_simple_2', trainer='lstm', epochs=1000, batch_size=50)
    elif c == 2:
        predicting('model_simple', test_bx[0], test_by[0])
    elif c == 3:
        training('model_conv', trainer='conv', epochs=1000, batch_size=50)
    elif c == 4:
        predicting('model_conv', test_bx[0], test_by[0])
    elif c == 5:
        training('model_state', trainer='stateful_lstm', epochs=100, batch_size=1)

    elif c == 6:
        training('model_FD001', trainer='lstm_by_id', epochs=200, batch_size=100)
    elif c == 7:
        x, y = get_shaped_array(test_xuid, test_yuid, 5)
        predicting('model_FD001', x, y)


if __name__ == '__main__':
    main(7)
