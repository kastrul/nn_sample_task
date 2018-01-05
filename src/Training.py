from matplotlib import pyplot

from ModelCreation import save_model


def plot_train_history(model):
    # plot history
    pyplot.plot(model.history['loss'], label='train')
    pyplot.plot(model.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


def stateful_lstm_trainer(train_x, train_y, test_x, test_y, model, model_name, epochs, batch_size):
    print('Training')
    for i in range(epochs):
        print('Epoch ', i + 1, '/', epochs)
        model.fit(train_x,
                  train_y,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=2,
                  validation_data=(test_x, test_y),
                  shuffle=False)
        model.reset_states()
    save_model(model, model_name)


def lstm_trainer(train_x, train_y, test_x, test_y, model, model_name, epochs=50, batch_size=100):
    # fit network
    history = model.fit(train_x,
                        train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(test_x, test_y),
                        verbose=2,
                        shuffle=False)

    save_model(model, model_name)
    plot_train_history(history)


def regular_trainer(train_x, train_y, test_x, test_y, model, model_name):
    # fit network
    history = model.fit(train_x,
                        train_y.reshape(train_y.shape[0], 1, 1),
                        epochs=70,
                        batch_size=60,
                        validation_data=(test_x, test_y.reshape(test_y.shape[0], 1, 1)),
                        verbose=2,
                        shuffle=True)

    save_model(model, model_name)
    plot_train_history(history)


def conv_trainer(train_x, train_y, test_x, test_y, model, model_name, epochs=50, batch_size=100):
    # fit network
    history = model.fit(train_x,
                        train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(test_x, test_y),
                        verbose=2,
                        shuffle=False)

    save_model(model, model_name)
    plot_train_history(history)
