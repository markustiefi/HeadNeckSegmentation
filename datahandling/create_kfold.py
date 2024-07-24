from sklearn.model_selection import KFold

def get_kfold(patlist, random_state = 1, n_splits = 5, fold = 0):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    for i, (train, test) in enumerate(kf.split(patlist)):
        if i == fold:
            return train, test