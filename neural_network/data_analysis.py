import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from neural_network.utils.load_dataset import load as load_dataset, label_to_text


def data_analysis():
    train, test = load_dataset()

    numbered_labels, train_class_count = np.unique(train.labels, return_counts=True)
    _, test_class_count = np.unique(test.labels, return_counts=True)
    translated_labels = translate_label_to_text(numbered_labels)

    train_info = pd.DataFrame([translated_labels, train_class_count], index=['labels', 'count']).T
    test_info = pd.DataFrame([translated_labels, test_class_count], index=['labels', 'count']).T

    print('plotting train dataset info')
    sns.barplot(data=train_info, x='labels', y='count',)
    plt.show()

    print('plotting test dataset info')
    sns.barplot(data=test_info, x='labels', y='count',)
    plt.show()


def translate_label_to_text(numbered_labels: list[int]):
    return list(map(lambda x: label_to_text[x], numbered_labels))

