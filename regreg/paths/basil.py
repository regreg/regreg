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
    unpen = np.zeros(path_obj.group_shape, np.bool)
    unpen[path_obj.unpenalized[1]] = True
    ever_active = list(path_obj.updated_ever_active(unpen)) # unpenalized groups

    solutions = []
    lagrange_solved = []

    all_failing = np.zeros(path_obj.group_shape, np.bool)

    M = min(step_nvar, (inactive_ranks >= 0).sum())

    candidate_bool = np.zeros(path_obj.group_shape, np.bool)
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
        
