import random
from copy import deepcopy

from dpipe.split import train_val_test_split, train_test_split, stratified_train_val_test_split, indices_to_ids, \
    kfold_split, train_test_split_groups


# ###########################################################
# ### backward compatibility with develop branch of dpipe ###

def _split_train(splits, val_size, groups=None, **kwargs):
    new_splits = []
    for train_val, test in splits:
        sub_groups = None if groups is None else groups[train_val]
        train, val = train_test_split_groups(
            train_val, val_size=val_size, groups=sub_groups, **kwargs) if val_size > 0 else (train_val, [])
        new_splits.append([train, val, test])
    return new_splits


def _train_val_test_split(ids, *, val_size, n_splits, random_state=42):
    split_indices = kfold_split(subj_ids=ids, n_splits=n_splits, random_state=random_state)
    split_indices = _split_train(splits=split_indices, val_size=val_size, random_state=random_state)
    return indices_to_ids(split_indices, ids)

# ### backward compatibility with develop branch of dpipe ###
# ###########################################################


def one2all(df, val_size=2, seed=0xBadCafe):
    """Train on 1 domain, test on (n - 1) domains."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        test_ids = df[df.fold != f].index.tolist()
        train_ids, val_ids = train_test_split(idx_b, test_size=val_size, random_state=seed) if val_size > 0 else \
                             (idx_b, [])
        split.append([train_ids, val_ids, test_ids])
    return split


def single_cv(df, n_splits=3, val_size=2, seed=0xBadCafe):
    """Cross-validation inside every domain."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        # cv_b = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)  # dpipe dev only
        cv_b = _train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)
        for cv in cv_b:
            split.append(cv)
    return split


def one2one(df, val_size=2, n_add_ids=5, train_on_add_only=False, seed=0xBadCafe, train_on_source_only=False):
    random.seed(seed)
    folds = sorted(df.fold.unique())
    split = []
    for fb in folds:
        folds_o = set(folds) - {fb}
        ids_train = [] if train_on_add_only else df[df.fold == fb].index.tolist()

        for fo in folds_o:
            ids_test, ids_train_add = train_test_split(df[df.fold == fo].index.tolist(), test_size=n_add_ids,
                                                       random_state=seed) if n_add_ids > 0 else \
                                          (df[df.fold == fo].index.tolist(), [])

            split.append([deepcopy(ids_train), random.sample(ids_test, val_size), ids_test]
                         if (not train_on_add_only) and train_on_source_only else
                         [deepcopy(ids_train) + ids_train_add, random.sample(ids_test, val_size), ids_test])
    return split


def strat_cv_debug(df, n_splits=3, seed=0xBadCafe):
    train_val_test_ids = stratified_train_val_test_split(df.index.tolist(), df.fold.tolist(),
                                                         val_size=0, n_splits=n_splits, random_state=seed)
    train_val_test_ids = [[e[0], e[2], e[2]] for e in train_val_test_ids]
    return train_val_test_ids
