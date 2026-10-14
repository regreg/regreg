"""
The BASIL (batch screening iterative lasso) path algorithm.
"""

import numpy as np
from ..affine import power_L

def basil_inner_loop(path_obj,
                     lagrange_subseq,
                     initial_data, # (solution, grad) pair
                     inner_tol=1.e-5,
                     verbose=False,
                     initial_step=None,
                     check_active=False,
                     step_nvar=50,
                     candidate_set=None,
                     solve_args={}):
    """
    Solve a path over a short sequence of Lagrange parameters.

    The subproblem is restricted to a candidate set formed from the
    unpenalized groups, the `step_nvar` inactive groups with the largest KKT
    violations, and `candidate_set`. The loop stops at the first Lagrange
    parameter at which the KKT conditions fail.

    Parameters
    ----------
    path_obj : `regreg.paths.grouped_path`
        Path object describing the problem.
    lagrange_subseq : sequence of float
        Decreasing sequence of Lagrange parameters.
    initial_data : tuple
        Pair ``(solution, grad_solution)`` used as a warm start.
    inner_tol : float, optional
        Tolerance for each subproblem.
    verbose : bool, optional
        Print progress information?
    initial_step : float, optional
        Initial step size for the solver. If None, it is estimated from the
        Lipschitz constant of ``path_obj.X``.
    check_active : bool, optional
        Also check KKT conditions for active groups?
    step_nvar : int, optional
        Number of inactive groups to add to the candidate set.
    candidate_set : list, optional
        Extra groups to include in the subproblem.
    solve_args : dict, optional
        Extra arguments passed to ``path_obj.solve_subproblem``.

    Returns
    -------
    lagrange_solved : list
        Lagrange parameters for which a solution was found.
    solutions : list
        Corresponding solutions.
    last_grad : ndarray
        Gradient at the last solution found.
    ever_active : list
        Updated candidate set of groups.
    """

    debug = True
    coef_stop = False

    # take a guess at the inverse step size
    if initial_step is None:
        _lipschitz = power_L(path_obj.X, max_its=50)
        final_step = 1000. / _lipschitz 
    else:
        final_step = initial_step

    # gradient of restricted elastic net at lambda_max

    solution, grad_solution = initial_data

    solution = solution.copy()
    last_grad = grad_solution.copy()
    inactive_ranks = path_obj.check_KKT(grad_solution,
                                        solution,
                                        path_obj.alpha * np.min(lagrange_subseq))[1]

    linear_predictor = path_obj.X.dot(solution)
    unpen = np.zeros(path_obj.group_shape, bool)
    unpen[path_obj.unpenalized[1]] = True
    ever_active = list(path_obj.updated_ever_active(unpen)) # unpenalized groups

    solutions = []
    lagrange_solved = []

    all_failing = np.zeros(path_obj.group_shape, bool)

    M = min(step_nvar, (inactive_ranks >= 0).sum())

    candidate_bool = np.zeros(path_obj.group_shape, bool)
    subproblem_set = path_obj.updated_ever_active((inactive_ranks < M) * (inactive_ranks >= 0) +
                                                  unpen)
    if candidate_set is not None:
        subproblem_set = sorted(set(subproblem_set + candidate_set))

    for lagrange in lagrange_subseq:

        (final_step, 
         subproblem_grad, 
         subproblem_soln,
         subproblem_linpred,
         subproblem_vars) = path_obj.solve_subproblem(solution, # for warm start
                                                      subproblem_set,
                                                      lagrange,
                                                      tol=inner_tol,
                                                      start_step=final_step,
                                                      debug=debug and verbose,
                                                      coef_stop=coef_stop,
                                                      **solve_args)

        saturated_grad = path_obj.saturated_loss.smooth_objective(subproblem_linpred, 'grad')
        # as subproblem always contains ever active, 
        # rest of solution should be 0
        solution[subproblem_vars] = subproblem_soln

        enet_grad = path_obj.enet_grad(solution, 
                                       path_obj._penalized_vars,
                                       lagrange)
        grad_solution[:] = (path_obj.full_gradient(path_obj.saturated_loss, 
                                                   subproblem_linpred) + 
                            enet_grad)
        failing_A, failing_I_ranks = path_obj.check_KKT(grad_solution, 
                                                        solution, 
                                                        path_obj.alpha * lagrange)

        if check_active:
            all_failing = failing_A + (failing_I_ranks >= 0)
        else:
            all_failing = failing_I_ranks >= 0

        if not all_failing.sum():
            ever_active_incr = path_obj.updated_ever_active(path_obj.active_set(solution))
            ever_active = sorted(set(ever_active + ever_active_incr))
            linear_predictor[:] = subproblem_linpred
            solutions.append(solution.T.copy())
            last_grad = grad_solution.copy()
            lagrange_solved.append(lagrange)
        else:
            ever_active1 = ever_active + list(subproblem_set)
            ever_active2 = path_obj.updated_ever_active(all_failing) 
            ever_active = sorted(set(ever_active1 + ever_active2))
            break

    return (lagrange_solved, 
            solutions, 
            last_grad,
            ever_active)

def basil(path_obj,
          lagrange_seq,
          initial_data, # (solution, grad) pair
          inner_tol=1.e-5,
          verbose=False,
          initial_step=None,
          check_active=False,
          step_nvar=50,
          step_lagrange=5,
          solve_args={}):
    """
    Compute a regularization path with the BASIL algorithm.

    The path is computed in batches of `step_lagrange` Lagrange parameters
    using `basil_inner_loop`.

    Parameters
    ----------
    path_obj : `regreg.paths.grouped_path`
        Path object describing the problem.
    lagrange_seq : ndarray
        Decreasing sequence of Lagrange parameters.
    initial_data : tuple
        Pair ``(solution, grad_solution)`` at ``lagrange_seq[0]``.
    inner_tol : float, optional
        Tolerance for each subproblem.
    verbose : bool, optional
        Print progress information? (Currently unused.)
    initial_step : float, optional
        Initial step size for the solver. (Currently unused.)
    check_active : bool, optional
        Also check KKT conditions for active groups? (Currently unused.)
    step_nvar : int, optional
        Number of inactive groups added to the candidate set in each batch.
    step_lagrange : int, optional
        Number of Lagrange parameters in each batch.
    solve_args : dict, optional
        Extra arguments passed to ``path_obj.solve_subproblem``.

    Returns
    -------
    solutions : ndarray
        Solutions along the path.
    """

    lagrange_solved, solutions, candidate_set = [np.inf], [], []

    cur_data = initial_data

    while True:
        
        lagrange_start = np.sum(lagrange_seq >= np.min(lagrange_solved))
        lagrange_cand = lagrange_seq[lagrange_start:(lagrange_start + step_lagrange)] 

        if len(lagrange_cand) > 0:
            (lagrange_incr, 
             solution_incr, 
             last_grad,
             candidate_set) = basil_inner_loop(path_obj,
                                               lagrange_cand,
                                               cur_data,
                                               inner_tol=inner_tol,
                                               step_nvar=step_nvar,
                                               candidate_set=candidate_set,
                                               solve_args=solve_args)
            if len(solution_incr) > 0:
                cur_soln = (solution_incr[-1], last_grad)

                solutions.extend(solution_incr)
                lagrange_solved.extend(lagrange_incr)
        else:
            break

    return np.array(solutions)
        
