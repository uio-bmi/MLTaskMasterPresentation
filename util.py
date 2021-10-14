import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
    encoder = OneHotEncoder(sparse=False)
    encoded_data = []
    for index, sequence in enumerate(data['sequence'].values):
        if index == 0:
            encoded_sequence = encoder.fit_transform(np.array(list(sequence)).reshape(-1, 1))
        else:
            encoded_sequence = encoder.transform(np.array(list(sequence)).reshape(-1, 1))
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
