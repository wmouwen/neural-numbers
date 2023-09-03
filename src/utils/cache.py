import os
import pathlib
import pickle

CACHE_DIR = f"{pathlib.Path(__file__).parent.parent.parent}/.cache"
STATE_FILE = "state"


def create_cache_directory():
    try:
        os.makedirs(CACHE_DIR)
    except FileExistsError:
        # directory already exists
        pass


def read_state():
    create_cache_directory()

    if not os.path.exists(f"{CACHE_DIR}/{STATE_FILE}"):
        return None

    with open(f"{CACHE_DIR}/{STATE_FILE}", 'rb') as filehandle:
        return pickle.load(filehandle)


def write_state(data):
    create_cache_directory()
    with open(f"{CACHE_DIR}/{STATE_FILE}", 'wb') as filehandle:
        pickle.dump(data, filehandle)
