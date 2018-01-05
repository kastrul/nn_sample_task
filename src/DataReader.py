import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import figaspect
from numpy.ma import ceil
from sklearn.preprocessing import MinMaxScaler


def ignore_columns(df, ignore):
    header = df.columns.values
    return [value for value in header if value not in ignore]


def plotter(array_x, array_y):
    fig = plt.figure()
    array_x = array_x.reshape(array_x.shape[0], array_x.shape[2])
    rows = ceil(array_x.shape[1] / 5)
    y = array_y
    print(array_x.shape)
    for col in range(array_x.shape[1] - 1):
        ax = fig.add_subplot(rows, 5, col + 1)
        x = array_x[:, col + 1]
        ax.set_ylim(0, x.max())
        ax.scatter(y, x, marker='.')
    plt.show()


def df_plotter(df, plot_nr):
    plt.rc('font', size=5)  # controls default text sizes
    plt.rc('axes', titlesize=5)  # fontsize of the axes title
    plt.rc('axes', labelsize=5)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the tick labels
    title = df['dataset_id'].values[0] + ' ' + str(df['unit_id'].values[0])
    w, h = figaspect(9.0 / 16.0)
    fig = plt.figure(figsize=(w, h), dpi=1000)
    fig.suptitle(title)
    headers = df.columns.values
    i = 1
    for header in headers:
        if header not in ['dataset_id', 'unit_id', 'cycle']:
            ax = fig.add_subplot(6, 5, i)
            df.plot.scatter(x='cycle', y=header, ax=ax, marker='.', s=1)
            i += 1
    fig.subplots_adjust(wspace=0.5,
                        hspace=0.8,
                        top=0.9,
                        bottom=0.0
                        )
    # plt.show()
    fig.savefig(r'../plots/plot_test_{0}.png'.format(str(plot_nr)))


def create_plots(df_array):
    df_count = len(df_array)
    plot_count = int(df_count / 50)
    for i in range(plot_count):
        try:
            df_plotter(df_array[i * 50], i)
        except IndexError:
            print('Created {0} plots.'.format(str(i - 1)))


def read_file(filename, columns=None):
    location = r'../data/' + filename
    dataframe = pd.read_csv(location, usecols=columns)

    return dataframe


def add_output_column(df, data_type):
    grouped = df.groupby(['dataset_id', 'unit_id'])
    if data_type == 'train':
        df = grouped.apply(lambda x: output_to_single_df(x))
    elif data_type == 'test':
        df = grouped.apply(lambda x: output_to_single_test(x))
    return df


def output_to_single_test(df):
    dataset_id = df['dataset_id'].values[0]
    unit_id = df['unit_id'].values[0]
    rd = read_file('RUL.csv')
    rul = 0
    try:
        rul = rd['rul'][(rd.dataset_id == dataset_id) & (rd.unit_id == unit_id)].values[0]
    except IndexError:
        print(dataset_id)
        print(unit_id)
    df['output'] = df['cycle'].values[::-1] + rul - 1
    return df


def output_to_single_df(df):
    df['output'] = df['cycle'].values[::-1] - 1
    return df


def get_data_frames():
    train_data = read_file('train.csv')
    test_data = read_file('test.csv')
    train_data = add_output_column(train_data, 'train')
    test_data = add_output_column(test_data, 'test')
    return train_data, test_data


def get_ndarrays():
    train, test = get_data_frames()
    train_x, train_y = train.values[:, 2:-1].astype('float32'), train.values[:, -1].astype('float32')
    test_x, test_y = test.values[:, 2:-1].astype('float32'), test.values[:, -1].astype('float32')
    return train_x, train_y, test_x, test_y


def get_scaled_arrays(train, test):
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    train_x, train_y = train.values[:, 2:-1].astype('float32'), train.values[:, -1].astype('float32')
    test_x, test_y = test.values[:, 2:-1].astype('float32'), test.values[:, -1].astype('float32')

    shape_train = train_x.shape[0], 1, train_x.shape[1]
    shape_test = test_x.shape[0], 1, test_x.shape[1]

    return scaler_train.fit_transform(train_x).reshape(shape_train), train_y, \
           scaler_test.fit_transform(test_x).reshape(shape_test), test_y


def get_unit_dataframes():
    train, test = get_data_frames()

    train_batches = []
    grouped_train = train.groupby(['dataset_id', 'unit_id'])
    for name, group in grouped_train:
        train_batches.append(group)

    test_batches = []
    grouped_test = test.groupby(['dataset_id', 'unit_id'])
    for name, group in grouped_test:
        test_batches.append(group)

    return train_batches, test_batches


def get_data_by_id(dataset_id, train, test, ignore=None):
    train_by_id = train[train['dataset_id'] == dataset_id]
    test_by_id = test[test['dataset_id'] == dataset_id]

    if ignore is not None:
        columns = ignore_columns(train, ignore)
        train_by_id = train_by_id[columns]
        test_by_id = test_by_id[columns]

    train_x, train_y, test_x, test_y = get_scaled_arrays(train_by_id, test_by_id)
    return train_x, train_y, test_x, test_y


def get_unit_by_id(dataset_id, unit_id, train, test, ignore=None):
    train_by_id = train[(train['dataset_id'] == dataset_id) & (train['unit_id'] == unit_id)]
    test_by_id = test[(test['dataset_id'] == dataset_id) & (test['unit_id'] == unit_id)]

    if ignore is not None:
        columns = ignore_columns(train, ignore)
        train_by_id = train_by_id[columns]
        test_by_id = test_by_id[columns]

    train_x, train_y, test_x, test_y = get_scaled_arrays(train_by_id, test_by_id)
    return train_x, train_y, test_x, test_y


def get_unit_batches(train, test):
    train_batches_x = []
    train_batches_y = []
    grouped_train = train.groupby(['dataset_id', 'unit_id'])
    for name, group in grouped_train:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_x = scaler.fit_transform(group.values[:, 2:-1].astype('float32'))
        shape = train_x.shape[0], 1, train_x.shape[1]

        train_batches_x.append(train_x.reshape(shape))
        train_batches_y.append(group.values[:, -1].astype('float32'))

    test_batches_x = []
    test_batches_y = []
    grouped_test = test.groupby(['dataset_id', 'unit_id'])
    for name, group in grouped_test:
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_x = scaler.fit_transform(group.values[:, 2:-1].astype('float32'))
        shape = test_x.shape[0], 1, test_x.shape[1]

        test_batches_x.append(test_x.reshape(shape))
        test_batches_y.append(group.values[:, -1].astype('float32'))

    return train_batches_x, train_batches_y, test_batches_x, test_batches_y


def main():
    # train_x, train_y, test_x, test_y = get_scaled_arrays()
    # train_batches_x, train_batches_y, test_batches_x, test_batches_y = get_unit_batches()

    # testing = pad_sequences(train_x, dtype='float32')
    # print(testing.shape)

    # plotter(train_batches_x[0], train_batches_y[0])
    # df_plotter(test_batches[100])

    # train_batches, test_batches = get_unit_dataframes()
    # create_plots(test_batches)

    train_x, train_y, test_x, test_y = get_data_by_id('FD004')
    print(train_x.shape)


if __name__ == '__main__':
    main()
