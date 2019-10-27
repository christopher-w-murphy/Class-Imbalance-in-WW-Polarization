import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from time import time


def results_dict():
    return {
        'fit_time': np.array([]),
        'score_time': np.array([]),
        'test_roc_auc': np.array([]),
        'test_average_precision': np.array([]),
        'test_roc_curve': np.array([]),
        'test_precision_recall_curve': np.array([])
    }


def model_analysis(model, results, X, y, tr_idx, te_idx, early_stopping=False, **kwargs):
    """
    This early stopping argument is for tree based models in scikit-learn.
    For early stopping in Keras use its callback argument.
    """
    try:
        if clf:
            del clf
    except NameError:
        pass
    clf = model

    X_tr = X[tr_idx]
    X_te = X[te_idx]
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    t0 = time()
    if early_stopping:
        best_loss = 1.0 * 10**6
        for _ in range(100):
            clf.fit(X_tr, y_tr)
            current_loss = log_loss(y_tr, clf.predict_proba(X_tr).T[1])
            if current_loss < best_loss:
                best_loss = current_loss
                clf.n_estimators += 10
            else:
                break
    else:
        clf.fit(X_tr, y_tr, **kwargs)
    results['fit_time'] = np.append(results['fit_time'], time() - t0)

    t1 = time()
    probas = clf.predict_proba(X_te)
    if probas.shape[1] == 2:
        probas = probas.T[1]
    results['test_roc_auc'] = np.append(results['test_roc_auc'], roc_auc_score(y_te, probas))
    results['test_average_precision'] = np.append(results['test_average_precision'], average_precision_score(y_te, probas))
    results['test_roc_curve'] = np.append(results['test_roc_curve'], roc_curve(y_te, probas))
    results['test_precision_recall_curve'] = np.append(results['test_precision_recall_curve'], precision_recall_curve(y_te, probas))
    results['score_time'] = np.append(results['score_time'], time() - t1)
