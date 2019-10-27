import pickle

def pkl_save_obj(obj, file_path):
    with open(file_path, 'wb') as f:
        if file_path.endswith('pkl'):
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pkl_load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
