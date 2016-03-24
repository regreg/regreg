
## Logistic regression with regreg

The logistic loss is included in regreg, taking a pair $(X,Y)$ where $X$ is an affine_transform and $Y$ is either a binary vector or, if not, an additional argument of 'trials' is needed specifying how many trials
per row of $Y$. This is equivalent to setting the weights option in R.


```python
import numpy as np, regreg.api as rr

n, p = 70, 8
Xr = np.random.standard_normal((n, p))
Yr = np.random.binomial(1,0.5, (n,))
lossr = rr.logistic_loss(Xr,Yr)
coefsr = lossr.solve(coef_stop=True, tol=1.e-8)

```

Let's compare this with R's output. First, we load the R magic into ipython


```python
%load_ext rmagic
cr = %R -i Xr -i Yr mylm = glm(Yr~Xr-1, family=binomial()); mylm$coef
print np.linalg.norm(coefsr-cr) / np.linalg.norm(cr)

```

Now, let's compare the objective value. By default, regreg divides the deviance by n. This can be changed by adding the argument "coef" to logistic loss which multiplies the objective by coef.


```python
print lossr.smooth_objective(coefsr, 'func') * n
fromR = %R summary(mylm)$deviance

loss2 = lossr = rr.logistic_loss(Xr,Yr, coef=n)
print loss2.smooth_objective(coefsr, 'func'), fromR

```

We can also check that using trials is equivalent to using weights. _This is obviously a bug_


```python
trials = np.ones(n, np.float)
trials[:20] = 2
trials[20:30] = 3.

lossw = rr.logistic_loss(Xr,Yr,trials=trials, coef=n)
coefsw = lossw.solve(tol=1.e-8)
%R -i trials
coefsRw = %R glm(Yr ~ Xr - 1, weights=trials, family=binomial())$coef
print coefsw.shape
print np.linalg.norm(coefsw-coefsRw) / np.linalg.norm(coefsw)
print lossw.smooth_objective(coefsw, 'func'), lossw.smooth_objective(coefsRw, 'func')
%R summary(glm(Yr~Xr-1, weights=trials, family=binomial()))$deviance
```

## Data import for newsgroup example

We will compare *regreg*'s solution to the newsgroup data to *glmnet*. The dataset is large with $(n,p)$=(11314, 777811)   though the design matrix is sparse. Unfortunately, we can't load in sparse matrices directly from R. So, we will write out to .csv save as .mat and use scipy.io.loadmat.


```python
import urllib, os, scipy.io, scipy.sparse

if not os.path.exists('newsgroup.mat'):
    print 'Had to form "newsgroup.mat"'
    if not os.path.exists('NewsGroup.RData'):
        print 'Had to download the data....'
        with file('NewsGroup.RData', 'w') as f:
            f.write(urllib.urlopen('http://www.jstatsoft.org/v33/i01/supp/6').read())
  
    %R library(Matrix)
    %R load('NewsGroup.RData')
    %R newsX = NewsGroup$x
    %R newsY = NewsGroup$y
    %R writeMM(newsX, 'newsX.mtx')
        
    X = scipy.io.mmread('newsX.mtx')
    Y = %R newsY 
    scipy.io.savemat('newsgroup.mat', {'X':X, 'Y':Y})
    


```

## Loss and penalty specification

Our loss function assumes binary successes (or proportions in [0,1]). We will center and scale our design matrix after having added an intercept to it. By default, scale and center are True for normalize while
intercept_column defaults to None. Finally, our loss is logistic loss.


```python
D = scipy.io.loadmat('newsgroup.mat')
X = D['X']; Y = D['Y']

# convert to binary
Y = (Y + 1) / 2; Y.shape = -1

# add intercept and normalize
X1 = scipy.sparse.hstack([np.ones((X.shape[0], 1)), X]).tocsc() # we use csc because we slice columns later
Xn = rr.normalize(X1, center=True, scale=True, intercept_column=0)
n, p = Xn.output_shape[0], Xn.input_shape[0]

# form the loss

loss = rr.logistic_loss(Xn, Y, coef=0.5)
print n, p
```

At this point, we are ready to define the penalty: the LASSO penalty, what else? We'll also compute $\lambda_{\max}$ the smallest value for which all coefficients are 0. This is the $\ell_{\infty}$ norm of the 
gradient at 0, almost. Actually, it should really be the $\ell_{\infty}$ norm of the gradient at the null model, i.e. with just an intercept. However, at an optimal solution the coordinate of the gradient corresponding to the
intercept must be 0 (in fact for any value of $\lambda$) so only the penalized columns enter.


```python
weights=np.ones(p); weights[Xn.intercept_column] = 0;
coefs = np.zeros(p)

lagrange_max = np.fabs(loss.smooth_objective(coefs, 'grad'))[1:].max()
penalty = rr.weighted_l1norm(weights, lagrange=0.7*lagrange_max)
lipschitz = rr.power_L(Xn) / n

print lipschitz, lagrange_max, loss.coef
```

As mentioned above, this null model does not change $\lambda_{\max}$.


```python

null_design = np.ones((n,1))
null_loss = rr.logistic_loss(null_design, Y)
null_coef = null_loss.solve()


```


```python
coefs[0] = null_coef
lagrange_max_null = np.fabs(loss.smooth_objective(coefs, 'grad'))[1:].max()
print lagrange_max_null, lagrange_max
```

The problem is a simple problem, in the sense that its prox is separable so we can instantiate it as:


```python
%%timeit -n 1 -r 1
problem = rr.simple_problem(loss, penalty)
problem.lipschitz = lipschitz
soln = problem.solve(start_step=lipschitz/1000, debug=True, tol=1.e-9)
loss.objective(soln)
print 'Final objective value: ', problem.objective(soln)
```

Another way to do it, which will turn out to be slightly easier for maintaining the so-called "active" set is to think of the penalty on the linear coefficients as one penalty and 
the "zero" penalty on the intercept as a separate penalty. The problem is still separable, we will just explicitly state is as separable with *only* a problem on the linear coefficients. It takes roughly the same number of iterations to solve. We won't use %%timeit because we want to store the output for use in demonstrating the strong rules.


```python

linear_slice = slice(1, Xn.input_shape[0])
linear_penalty = rr.l1norm(p-1, lagrange=0.7*lagrange_max)
separable = rr.separable_problem(loss, Xn.input_shape, [linear_penalty], [linear_slice])
separable.coefs[0] = null_coef
final_inv_step = lipschitz / 1000

separable_soln = separable.solve(start_step=final_inv_step, tol=1.e-9,debug=True)
print 'Final objective value: ', separable.objective(separable_soln)
```

## Strong rules

The strong rules take a current solution, with its active variables try to guess which variables will enter the model at a new value of the 
Lagrange parameter. It does this by guessing a bound on the slope of the dual paths. Typically this value is 1, but it could be other values as well.

In this example, we will use the so-called strong rules (see http://arxiv.org/pdf/1011.2234.pdf) to screen variables at each step.
The rule takes the gradient of the smooth part of the problem at $\lambda_{\text{cur}}$ and tries to guess which variables will still be excluded at $\lambda_{\text{new}} < \lambda_{\text{cur}}$.
There are also possibly some unpenalized columns, these are ignored as the KKT conditions say that those entries of the gradient must be 0 at a minimizer.


```python
def strong_set_lasso(grad, penalized, lagrange_cur, lagrange_new, slope_estimate=1):
    if not isinstance(penalized, rr.selector):
        s = rr.selector(penalized, grad.shape)
    else:
        s = penalized
    value = np.zeros(grad.shape, np.bool)
    value += (s.adjoint_map(np.fabs(s.linear_map(grad)) < (slope_estimate+1) \
                                * lagrange_new - slope_estimate*lagrange_cur) >= 0)
    return ~value

g = loss.smooth_objective(separable_soln, 'grad')
linear_selector = separable.selectors[0]
strong_set = strong_set_lasso(g, linear_selector, 0.7 * lagrange_max, 0.6 * lagrange_max)
print strong_set
```

Knowing a strong set means the problem can be solved much faster.




```python
def restricted_problem(Xn, Y, candidate_set, lagrange):
    '''
    Assumes the candidate set includes intercept as first column.
    '''
    print candidate_set.sum()
    Xslice = Xn.slice_columns(candidate_set)
    Xslice.intercept_column = 0
    loss = rr.logistic_loss(Xslice, Y, coef=0.5)
    linear_slice = slice(1, Xslice.input_shape[0])
    linear_penalty = rr.l1norm(Xslice.input_shape[0]-1, lagrange=lagrange)
    candidate_selector = rr.selector(candidate_set, Xn.input_shape)
    penalized_selector = rr.selector(candidate_set[1:], Xn.input_shape)
    problem_sliced = rr.separable_problem(loss, Xslice.input_shape, [linear_penalty], [linear_slice])
    return problem_sliced, candidate_selector, penalized_selector

%timeit sp = restricted_problem(Xn, Y, strong_set, 0.6 * lagrange_max)[0]; sp.solve(start_inv_step=lipschitz / 1000, tol=1.e-9)

subproblem, strong_selector, penalized_selector = restricted_problem(Xn, Y, strong_set, 0.6 * lagrange_max)
sub_soln = subproblem.solve(start_inv_step=lipschitz / 1000, tol=1.e-9)

```

Compare this to one solution using all coefficients.



```python
%%timeit -n 1 -r 1

linear_penalty.lagrange = 0.6 * lagrange_max 
separable = rr.separable_problem(loss, Xn.primal_shape, [linear_penalty], [linear_slice])
separable.coefs[0] = null_coef
final_inv_step = lipschitz / 1000

separable_soln[:] = separable.solve(start_inv_step=final_inv_step, tol=1.e-9,debug=True)

```

In this case, we see that solving the problem with all the coefficients whose nonzero coefficients are contained in the strong set. Hence we know that solving the problem with fewer coefficients actually solves the bigger problems.


```python
set(np.nonzero(separable_soln != 0)[0]).issubset(strong_set)

```

Generally, we have to check whether solving the problem on the candidate strong set and setting the rest to zero
 solves the entire problem. In other words, we need to check the KKT conditions are satisfied.
If we let  be the solution that is 0 everywhere outside the strong set and
 agrees with the subproblem on the strong set, then we must check that

\begin{eqnarray}
|\nabla L(\hat{\beta}_{\text{sub}})_i| &< \lambda \quad & i  \in \text{strong}^c \cap \text{penalized} \\
\nabla L(\hat{\beta}_{\text{sub}})_i &=\lambda \sgn(\nabla L(\hat{\beta}_{\text{sub}})_i)& i  \in \text{strong} \cap \text{penalized} \\
\nabla L(\hat{\beta}_{\text{sub}})_i &= 0 & i  \in \text{strong} \cap \text{penalized}^c \\
\end{eqnarray}


In this function, however, we assume that the unpenalized coefficients have been solved sufficiently and only check the penalized ones.



```python
def check_KKT(grad, penalized, solution, lagrange, tol=1.0e-02):
    '''
    Verify that the KKT conditions for the LASSO possibly with unpenalized coefficients
    is satisfied for (grad, solution) where grad is the gradient of the loss evaluated
    at solution.
    '''
    if not isinstance(penalized, rr.selector):
        s = rr.selector(penalized, grad.shape)
    else:
        s = penalized
    soln_s = s.linear_map(solution)
    g_s = s.linear_map(grad)
    failing_s = np.zeros(g_s.shape)
    failing = np.zeros(grad.shape, np.bool)

    # Check the inactive coefficients
    failing += s.adjoint_map(np.fabs(g_s) > lagrange * (1 + tol))

    # Check the active coefficients
    active = soln_s != 0
    failing_s[active] += np.fabs(g_s[active] / lagrange + np.sign(soln_s[active])) >= tol 
    failing += s.adjoint_map(failing_s)
    return failing

expanded_soln = strong_selector.adjoint_map(sub_soln)
full_grad = loss.smooth_objective(expanded_soln, 'grad')
failing = check_KKT(full_grad, penalized_selector, expanded_soln, linear_penalty.lagrange)
print failing.sum()
```

We finally have the makings of a full algorithm using the strong rules. Given a candidate
 active set, we solve the restricted problem


```python
def fit(grad_cur, penalized, Xn, Y, soln_cur, lagrange_cur, lagrange_new, active, start_inv_step=1, tol=1.e-6, slope_estimate=2, coef_stop=True, debug=False):
    
    # try to solve the problem with the active set
    active_subproblem, active_selector, pen_active_selector = restricted_problem(Xn, Y, active, lagrange_new)
    active_subproblem.coefs[:] = active_selector.linear_map(soln_cur)
    active_soln = active_subproblem.solve(start_inv_step=start_inv_step, coef_stop=coef_stop, tol=tol, debug=debug)
    soln_cur[:] = active_selector.adjoint_map(active_soln)
        
    final_inv_step = active_subproblem.final_inv_step
    strong = strong_set_lasso(grad_cur, penalized, lagrange_cur, lagrange_new, slope_estimate=slope_estimate)
    #strong_subproblem, strong_selector, pen_strong_selector = restricted_problem(Xn, Y, strong, lagrange_new)
    #strong_subproblem.coefs[:] = strong_selector.linear_map(soln_cur)
    
    #grad_cur = strong_selector.adjoint_map(strong_subproblem.smooth_objective(strong_subproblem.coefs, 'grad'))
    #soln_cur[:] = strong_selector.adjoint_map(strong_subproblem.coefs)
    # are any strong coefficients failing?
    #strong_failing = check_KKT(grad_cur,
    #                           pen_strong_selector,
    #                           soln_cur,
    #                           lagrange_new)
    #if strong_failing.sum():
    #    failing = strong_selector.adjoint_map(strong_failing)
    #    return failing + active, final_inv_step, strong
    return soln_cur != 0, final_inv_step, strong



```


```python
import gc
def main():

    lagrange_sequence = lagrange_max * np.exp(np.linspace(np.log(0.05), 0, 100))[::-1]

    # scaling will be needed to get coefficients on original scale                                                                                   \
                                                                                                                                                      

    scalings = np.asarray(Xn.col_stds).reshape(-1)
    final_inv_step = lipschitz / 1000
    # first solution corresponding to all zeros except intercept                                                                                     \
                                                                                                                                                      

    solution = np.zeros(p)
    solution[0] = null_coef

    penalized = np.ones(solution.shape, np.bool)
    penalized[0] = False
    grad = loss.smooth_objective(solution, 'grad')
    strong = strong_set_lasso(grad, penalized, lagrange_sequence[0], lagrange_sequence[1])
    active = strong.copy()

    solutions = [solution.copy()]
    rescaled_solutions = [solution.copy()[1:]]
    objective = [loss.smooth_objective(solution, 'func')]
    dfs = [1]
    retry_counter = 0
    import time
    toc = time.time()
    for lagrange_new, lagrange_cur in zip(lagrange_sequence[1:], lagrange_sequence[:-1]):
        num_tries = 0
        debug = False
        tol = 1.0e-5
        while True:
            active_new, final_inv_step, strong = fit(grad, penalized, Xn,
                                             Y, solution, lagrange_cur,
                                             lagrange_new, active,
                                             tol=tol,
                                             start_inv_step=final_inv_step,
                                             debug=debug)
            grad = loss.smooth_objective(solution, 'grad')
            if active_new.sum() <= active.sum() and (~active * active_new).sum() == 0:
                failing = check_KKT(grad, penalized, solution, lagrange_new)
                if not failing.sum():
                    active = (solution != 0) + active
                    break
                else:
                    retry_counter += 1
                    print 'trying again:', retry_counter, 'failing:', np.nonzero(failing)[0], active.sum()
                    active += strong
            else:
                print 'active set different', np.nonzero(active), np.nonzero(active_new)
                active = active_new + strong 
                
                
            tol /= 2.
            num_tries += 1
            if num_tries % 5 == 0:
                debug=True
                tol = 1.0e-5
                active += strong_set_lasso(grad, penalized, lagrange_cur, lagrange_new)

        solutions.append(solution.copy())
        rescaled_solutions.append(solution[1:] / scalings[1:])
        objective.append(loss.smooth_objective(solution, mode='func'))
        dfs.append(active.shape[0])
        print lagrange_cur / lagrange_max, lagrange_new, (solution != 0).sum(), 1. - objective[-1] / objective[0], list(lagrange_sequence).index(lagrange_new), np.fabs(rescaled_solutions[-1]).sum()
        gc.collect()
        tic = time.time()
    
        print 'time: %0.1f' % (tic-toc)
    solutions = scipy.sparse.lil_matrix(solutions)
    rescaled_solutions = scipy.sparse.lil_matrix(rescaled_solutions)
    objective = np.array(objective)
    output = {'devratio': 1 - objective / objective.max(),
              'df': dfs,
              'beta': solutions,
              'lagrange': lagrange_sequence,
              'scalings': scalings,
              'rescaled_beta': rescaled_solutions}
   
    scipy.io.savemat('newsgroup_results.mat', output)

main()
```

We finally define our problem, by restricting interest to only some columns.


```python
%%R
library(glmnet)
library(Matrix)
load("NewsGroup.RData")

newsX=NewsGroup$x
newsy=NewsGroup$y


x=newsX
y=(newsy+1)/2
n=nrow(newsX)
p=ncol(newsX)
```


```python
import time

toc = time.time()

%R a=glmnet(x,as.factor(y),stand=TRUE,family="binomial",lambda.min=0.05); print(a$names)

tic = time.time()
print 'R time: %0.1f ' % (tic-toc)
```


```python
%%R
newsgroupFit=list(a=a)
save(newsgroupFit,file="newsgroupFit.RData")
l1norm = apply(abs(a$beta), 2, sum)
write.table(data.frame(lagrange=a$lambda, devratio=a$dev.ratio, df=a$df, l1=l1norm), 'newsgroup_output.csv',row.names=FALSE, sep=',')


```


```python

```
