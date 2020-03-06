import numpy as np

from sklearn.model_selection import check_cv
from joblib.parallel import Parallel, delayed

have_sklearn = True
try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    have_sklearn = False

def cross_validate(path_fitter,
                   lagrange_seq,
                   cv=5,
                   group_labels=None,
                   n_jobs=-1,
                   score=lambda x,y: np.linalg.norm(x-y)**2 / y.shape[0],
                   verbose=True,
                   inner_tol=1.e-5):

    '''
    Cross validates a path fitter over a set of 
    Lagrange parameters.

    Adjusts coefficient of subsampled loss by fraction ntrain / nsample.

    Parameters
    ----------

    path_fitter : group_path
    '''

    # assumes that full data of path_fitter.saturated_loss
    # is the same as

    # Compute paths for all folds 

    response = path_fitter.saturated_loss.data
    nsample = response.shape[0]
    have_sklearn = False
    if have_sklearn:
        cv = check_cv(cv)
        folds = list(cv.split(range(nsample), range(nsample)))
    else:
        if type(cv) != type(3):
            raise ValueError('without sklearn, cv should be an integer')
        groups = np.floor(np.arange(nsample)  / nsample * cv).astype(np.int)
        np.random.shuffle(groups)
        folds = [(groups != g, groups == g)
                 for g in np.unique(groups)]

    # Loop over folds, computing path
    # for each (train, test)

    def fit_fold(path_fitter,
                 lagrange_seq,
                 test,
                 train,
                 inner_tol):
        train_path = path_fitter.subsample(train)
        train_results = train_path.main(lagrange_seq, inner_tol=inner_tol)
        return train_results

    jobs = (delayed(fit_fold)(path_fitter,
                              lagrange_seq,
                              test,
                              train,
                              inner_tol)
            for train, test in folds)

    # Execute the jobs in parallel using joblib
    results = Parallel(n_jobs=n_jobs, verbose=verbose,
                        backend="threading")(jobs)

    # Form the linear predictors and responses

    linpred_responses = [(path_fitter.linpred(result['beta'], 
                                              path_fitter.X,
                                              train_test[1]),
                          response[train_test[1]])
                         for train_test, result in zip(folds, results)]
    scores = [[score(linpred.T[i].T, y) for i in range(linpred.shape[-1])] for (linpred, y) in linpred_responses]

    return np.mean(scores, 0), np.std(scores, 0)

    

