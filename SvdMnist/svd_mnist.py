import tqdm
import numpy as np
from keras.datasets import mnist

(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Control how many "dominating" images to use.
top_k = 8

digit_samples = dict((i, None) for i in range(10))

for i in tqdm.tqdm(range(len(train_label)), ascii=True):
    if digit_samples[train_label[i]] is None:
        digit_samples[train_label[i]] = train_data[i].reshape((28 * 28, 1))
    else:
        digit_samples[train_label[i]] = np.concatenate(
            (digit_samples[train_label[i]], train_data[i].reshape(28 * 28, 1)), axis=1)

svd_decomposed = dict((i, None) for i in range(10))
for i in tqdm.tqdm(range(10), ascii=True):
    svd_decomposed[i] = np.linalg.svd(digit_samples[i])[0][:, :top_k]

del digit_samples

from matplotlib import pyplot as plt
f, g = plt.subplots(2, 4)
for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        g[i][j].imshow(svd_decomposed[6][:, idx].reshape((28, 28)), cmap='gray')
f.show()
plt.show()
exit()




accurate_count = 0
for i in tqdm.tqdm(range(len(test_data)), ascii=True):
    data, label = test_data[i], test_label[i]
    data = data.reshape((28 * 28, 1))
    min_residual = None
    predict = None
    for category in range(10):
        residual = np.linalg.lstsq(svd_decomposed[category], data)[1]
        if min_residual is None or residual < min_residual:
            min_residual = residual
            predict = category
    assert predict is not None
    if predict == label:
        accurate_count += 1

print(f'accuracy: {accurate_count / len(test_data)}')
