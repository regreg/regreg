"""
Cross-validation of regularization paths.
"""

import numpy as np

from joblib.parallel import Parallel, delayed

have_sklearn = True
try:
    from sklearn.model_selection import check_cv
except ImportError:
    have_sklearn = False
    pass

from . import strong_rules

def cross_validate(path_obj,
                   lagrange_seq,
                   path_fitter=strong_rules,
                   initial_data=None,
                   cv=5,
                   group_labels=None,
                   n_jobs=-1,
                   score=lambda x,y: np.linalg.norm(x-y)**2 / y.shape[0],
                   verbose=True,
                   inner_tol=1.e-5):

    '''
    Cross-validate a path fitter over a set of Lagrange parameters.

    The coefficient of each subsampled (training) loss is multiplied by
    ``nsample / ntrain``.

    Parameters
    ----------
    path_obj : `regreg.paths.grouped_path`
        Path object describing the problem on the full data.
    lagrange_seq : sequence of float
        Decreasing sequence of Lagrange parameters.
    path_fitter : callable, optional
        Function that computes the path, with signature like
        `regreg.paths.strong_rules`.
    initial_data : tuple, optional
        Pair ``(solution, grad_solution)`` to start each path. Defaults to
        ``(path_obj.solution, path_obj.grad_solution)``.
    cv : int, optional
        Number of folds.
    group_labels : ndarray, optional
        Currently unused.
    n_jobs : int, optional
        Number of jobs passed to `joblib.Parallel`.
    score : callable, optional
        Function of ``(linear_predictor, response)`` returning the score on
        a test fold. Defaults to mean squared error.
    verbose : bool, optional
        Verbosity passed to `joblib.Parallel`.
    inner_tol : float, optional
        Tolerance for each subproblem.

    Returns
    -------
    mean_score : ndarray
        Mean score over folds, for each Lagrange parameter.
    std_score : ndarray
        Standard deviation of score over folds, for each Lagrange parameter.
    '''

    # assumes that full data of path_obj.saturated_loss
    # is the same as

    # Compute paths for all folds 

    response = path_obj.saturated_loss.data
    nsample = response.shape[0]
    have_sklearn = False
    if have_sklearn:
        cv = check_cv(cv)
        folds = list(cv.split(range(nsample), range(nsample)))
    else:
        if type(cv) != type(3):
            raise ValueError('without sklearn, cv should be an integer')
        groups = np.floor(np.arange(nsample)  / nsample * cv).astype(int)
        np.random.shuffle(groups)
        folds = [(groups != g, groups == g)
                 for g in np.unique(groups)]

    # Loop over folds, computing path
    # for each (train, test)

    if initial_data is None:
        initial_data = (path_obj.solution,
                        path_obj.grad_solution)

    def fit_fold(path_obj,
                 lagrange_seq,
                 initial_data,
                 test,
                 train,
                 inner_tol):
        train_path = path_obj.subsample(train)
        train_path.saturated_loss.coef *= train.shape[0] / train.sum()
        train_results = path_fitter(train_path, 
                                    lagrange_seq, 
                                    initial_data,
                                    inner_tol=inner_tol)
        return train_results

    jobs = (delayed(fit_fold)(path_obj,
                              lagrange_seq,
                              initial_data,
                              test,
                              train,
                              inner_tol)
            for train, test in folds)

    # Execute the jobs in parallel using joblib
    results = Parallel(n_jobs=n_jobs,
                       verbose=verbose,
                       prefer="threads")(jobs)

    # Form the linear predictors and responses

    linpred_responses = [(path_obj.linpred(result['beta'], 
                                              path_obj.X,
                                              train_test[1]),
                          response[train_test[1]])
                         for train_test, result in zip(folds, results)]
    scores = [[score(linpred.T[i].T, y) for i in range(linpred.shape[-1])] for (linpred, y) in linpred_responses]

    return np.mean(scores, 0), np.std(scores, 0)


def cross_validate_alt(path_obj,
                       fit_fold,
                       fit_args,
                       lagrange_seq,
                       path_fitter=strong_rules,
                       initial_data=None,
                       cv=5,
                       group_labels=None,
                       n_jobs=-1,
                       score=lambda x,y: np.linalg.norm(x-y)**2 / y.shape[0],
                       verbose=True,
                       inner_tol=1.e-5):

    '''
    Cross-validate a path fitter over a set of Lagrange parameters.

    Like `cross_validate`, but with a user-supplied function to fit each
    fold.

    Parameters
    ----------
    path_obj : `regreg.paths.grouped_path`
        Path object describing the problem on the full data.
    fit_fold : callable
        Called as ``fit_fold(lagrange_seq, test, train, path_fitter,
        initial_data, *fit_args)``; must return a dict with key ``'beta'``.
    fit_args : sequence
        Extra positional arguments for `fit_fold`.
    lagrange_seq : sequence of float
        Decreasing sequence of Lagrange parameters.
    path_fitter : callable, optional
        Function that computes the path, passed to `fit_fold`.
    initial_data : tuple, optional
        Pair ``(solution, grad_solution)`` to start each path. Defaults to
        ``(path_obj.solution, path_obj.grad_solution)``.
    cv : int, optional
        Number of folds.
    group_labels : ndarray, optional
        Currently unused.
    n_jobs : int, optional
        Number of jobs passed to `joblib.Parallel`.
    score : callable, optional
        Function of ``(linear_predictor, response)`` returning the score on
        a test fold. Defaults to mean squared error.
    verbose : bool, optional
        Verbosity passed to `joblib.Parallel`.
    inner_tol : float, optional
        Currently unused.

    Returns
    -------
    mean_score : ndarray
        Mean score over folds, for each Lagrange parameter.
    std_score : ndarray
        Standard deviation of score over folds, for each Lagrange parameter.
    '''

    # assumes that full data of path_obj.saturated_loss
    # is the same as

    # Compute paths for all folds 

    response = path_obj.saturated_loss.data
    nsample = response.shape[0]
    have_sklearn = False
    if have_sklearn:
        cv = check_cv(cv)
        folds = list(cv.split(range(nsample), range(nsample)))
    else:
        if type(cv) != type(3):
            raise ValueError('without sklearn, cv should be an integer')
        groups = np.floor(np.arange(nsample)  / nsample * cv).astype(int)
        np.random.shuffle(groups)
        folds = [(groups != g, groups == g)
                 for g in np.unique(groups)]

    if initial_data is None:
        initial_data = (path_obj.solution,
                        path_obj.grad_solution)

    # Loop over folds, computing path
    # for each (train, test)

    jobs = (delayed(fit_fold)(lagrange_seq,
                              test,
                              train,
                              path_fitter,
                              initial_data,
                              *fit_args)
            for train, test in folds)

    # Execute the jobs in parallel using joblib
    results = Parallel(n_jobs=n_jobs,
                       verbose=verbose,
                       prefer="threads")(jobs)

    # Form the linear predictors and responses

    linpred_responses = [(path_obj.linpred(result['beta'], 
                                           path_obj.X,
                                           train_test[1]),
                          response[train_test[1]])
                         for train_test, result in zip(folds, results)]
    scores = [[score(linpred.T[i].T, y) for i in range(linpred.shape[-1])] for (linpred, y) in linpred_responses]

    return np.mean(scores, 0), np.std(scores, 0)



