import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ALPHABET = ["A", "C", "G", "T"]


def load_data(data_path: str) -> pd.DataFrame:
    sequences = []
    labels = []

    with open(data_path, "r") as file:
        for line in file.readlines():
            if ">" in line:
                labels.append(int(line.replace(">", "").replace("\n", "").replace(" ", "")))
            else:
                sequences.append(line.replace('\n', ''))

    assert all(isinstance(label, int) for label in labels)
    assert all(isinstance(seq, str) for seq in sequences)

    return pd.DataFrame({"sequence": sequences, "label": labels})


def encode_onehot(data: pd.DataFrame):
    encoded_data = []
    mapping = {letter: i for i, letter in enumerate(ALPHABET)}
    for index, sequence in enumerate(data['sequence'].values):
        encoded_sequence = np.eye(len(ALPHABET))[[mapping[letter] for letter in sequence]]
        encoded_data.append(encoded_sequence.flatten())

    feature_names = []
    for i in range(len(data['sequence'][0])):
        for j in range(len(ALPHABET)):
            feature_names.append(f"{ALPHABET[j]}_{i + 1}")

    return {"encoded_data": np.array(encoded_data), "labels": data["label"].values, "feature_names": feature_names}


def test_kmer_encoding(encode_kmer_frequencies_func):
    test_df = pd.DataFrame({'sequences': ['AAC', 'CAT', 'TAC', 'GCA'], 'labels': [1, 0, 1, 0]})

    encoded_test = encode_kmer_frequencies_func(dataset=test_df, k=2, learn_model=True)

    correct_encoded_data = np.array([[0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.],
                                     [0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0.]])

    assert 'labels' in encoded_test and isinstance(encoded_test['labels'], np.ndarray) and np.array_equal(encoded_test['labels'],
                                                                                                          np.array([1, 0, 1, 0]))

    assert set(encoded_test['feature_names']) == {'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT'}

    assert np.allclose(encoded_test['encoded_data'], correct_encoded_data), "There is an error in the encoded data."

    print("k-mer encoding works!")


def standardize(train_data: np.ndarray, test_data: np.ndarray):

    if set(np.unique(train_data)) != {0, 1}:
        scaler = StandardScaler()
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    else:
        return train_data, test_data
