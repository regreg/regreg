import numpy as np

from sklearn.model_selection import check_cv
from sklearn.metrics import mean_squared_error
from joblib.parallel import Parallel, delayed

def cross_validate(path_fitter,
                   lagrange_seq,
                   cv=5,
                   n_jobs=-1,
                   score=lambda x,y: np.linalg.norm(x-y)**2,
                   verbose=True,
                   inner_tol=1.e-5):

    # assumes that full data of path_fitter.saturated_loss
    # is the same as

    cv = check_cv(cv)

    # Compute paths for all folds 

    nsample = path_fitter.saturated_loss.shape[0]
    folds = list(cv.split(range(nsample), range(nsample)))

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

    response = path_fitter

    # Execute the jobs in parallel using joblib
    results = Parallel(n_jobs=n_jobs, verbose=verbose,
                        backend="threading")(jobs)

    # Form the linear predictors and responses
    response = path_fitter.saturated_loss.data

    linpred_responses = ((path_fitter.linpred(result['beta'], 
                                              path_fitter.X,
                                              train_test[1]).T,
                          response[train_test[1]])
                         for train_test, result in zip(folds, results))

    scores = [[score(yhat, y) for yhat in linpred] for (linpred, y) in linpred_responses]
    print(np.mean(scores, 0), np.std(scores, 0))
    return np.mean(scores, 0), np.std(scores, 0)

    

