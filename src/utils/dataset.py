import gzip
import numpy as np
import os
import struct
import urllib.request

DATA_DIR = os.path.join(os.getcwd(), '.cache', 'data')
REMOTE_BASE_URL = 'https://azureopendatastorage.blob.core.windows.net/mnist/'


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, 'train-images.gz')):
        urllib.request.urlretrieve(
            url=f"{REMOTE_BASE_URL}train-images-idx3-ubyte.gz",
            filename=os.path.join(DATA_DIR, 'train-images.gz')
        )
    if not os.path.exists(os.path.join(DATA_DIR, 'train-labels.gz')):
        urllib.request.urlretrieve(
            url=f"{REMOTE_BASE_URL}train-labels-idx1-ubyte.gz",
            filename=os.path.join(DATA_DIR, 'train-labels.gz')
        )
    if not os.path.exists(os.path.join(DATA_DIR, 'test-images.gz')):
        urllib.request.urlretrieve(
            url=f"{REMOTE_BASE_URL}t10k-images-idx3-ubyte.gz",
            filename=os.path.join(DATA_DIR, 'test-images.gz')
        )
    if not os.path.exists(os.path.join(DATA_DIR, 'test-labels.gz')):
        urllib.request.urlretrieve(
            url=f"{REMOTE_BASE_URL}t10k-labels-idx1-ubyte.gz",
            filename=os.path.join(DATA_DIR, 'test-labels.gz')
        )


def load(filename, label=False):
    with gzip.open(os.path.join(DATA_DIR, filename)) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))

        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]

            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)

        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)

    return res
