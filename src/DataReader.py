import pandas as pd


def get_data(filename, columns=None):
    location = r'../data/' + filename
    data_file = pd.read_csv(location, usecols=columns)
    return data_file


def add_output_column(df):
    grouped = df.groupby(['dataset_id', 'unit_id'])
    df = grouped.apply(lambda x: output_to_unique_dataframe(x))
    return df.reset_index(drop=True)


def output_to_unique_dataframe(df):
    max_cycle = df['cycle'].max()
    df = df.assign(output=lambda x: max_cycle - x.cycle)
    return df


def get_filtered_data(filename, data_type):
    data = get_data(filename)
    if data_type == 'train':
        data = add_output_column(data)
        data = data[get_columns(data, ['dataset_id', 'unit_id'])]
    return data


def get_training_in(df):
    data_array = df[get_columns(df, ['output'])].values
    return data_array


def get_training_out(df):
    return df['output'].values


def get_columns(df, ignore):
    header = df.columns.values
    return [value for value in header if value not in ignore]


def main():
    df_training = get_filtered_data('train.csv', 'train')
    training_in = get_training_in(df_training)
    training_out = get_training_out(df_training)
    print(training_in)
    print(training_out)


if __name__ == '__main__':
    main()
