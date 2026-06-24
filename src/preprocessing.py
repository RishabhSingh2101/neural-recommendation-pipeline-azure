 import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        print("Data loaded")
        return data

    def clean_data(self, data):
        data = data.dropna()
        data = data.drop_duplicates()
        return data

    def split_data(self, data):
        train, test = train_test_split(
            data,
            test_size=0.2,
            random_state=42
        )

        return train, test


if __name__ == "__main__":

    processor = DataPreprocessor(
        "data/ratings.csv"
    )

    data = processor.load_data()

    data = processor.clean_data(data)

    train, test = processor.split_data(data)

    print(train.shape)
    print(test.shape)
