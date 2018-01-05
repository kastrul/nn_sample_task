from ModelCreation import load_model


def lstm_predictor(test_x, model_name):
    model = load_model(model_name)
    model.compile(loss='mae', optimizer='adam')

    y_predict = model.predict(test_x)

    # print(y_predict)
    return y_predict
