import os
import pickle

CACHE_DIR = os.path.join(os.getcwd(), '.cache')
STATE_FILE = os.path.join(CACHE_DIR, 'state')


def read_state():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(STATE_FILE):
        return None

    with open(STATE_FILE, 'rb') as filehandle:
        return pickle.load(filehandle)


def write_state(data):
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(STATE_FILE, 'wb') as filehandle:
        pickle.dump(data, filehandle)
