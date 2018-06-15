from sklearn import manifold
from chainer.datasets import get_mnist
import pandas as pd
import seaborn as sns
import numpy as np


if __name__ == '__main__':
    tsne = manifold.TSNE(n_components=2)
    train, test = get_mnist()
    X, y = train._datasets

    size = 1000
    X = X[:size]
    y = y[:size]

    t = tsne.fit_transform(X)
    df = pd.DataFrame(np.array([t[:, 0], t[:, 1], y]).T, columns=['x', 'y', 'label'])
    sns.pairplot(df, hue='label')
    # sns.lmplot(df['x'], df['y'], data=df, hue='label').savefig('mnist.png')