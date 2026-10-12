"""
Cython wrappers for the Cox proportional hazards loss.

These wrap the C implementation (``cox_fns.c``) of the Cox negative log
partial likelihood, its gradient and Hessian-vector products.

All functions take the linear predictor together with preallocated work
buffers, case weights, censoring indicators, the ordering of the event
times and their 0-based ranks (with min and max tie breaking).
"""

import warnings
import numpy as np, cython
cimport numpy as cnp

DTYPE_float = float
ctypedef cnp.float_t DTYPE_float_t
DTYPE_int = int
ctypedef cnp.int_t DTYPE_int_t
ctypedef cnp.intp_t DTYPE_intp_t

cdef extern from "cox_fns.h":

    void _update_cox_exp(double *linear_pred_ptr, # Linear term in objective 
                         double *exp_ptr,         # stores exp(eta) 
                         double *exp_accum_ptr,   # inner accumulation vector 
                         double *case_weight_ptr, # case weights 
                         size_t *censoring_ptr,     # censoring indicator 
                         size_t *ordering_ptr,      # 0-based ordering of times 
                         size_t *rankmin_ptr,       # 0-based ranking with min tie breaking 
                         size_t ncase)              # how many subjects / times 

    void _update_cox_expZ(double *linear_pred_ptr,  # Linear term in objective 
                          double *right_vector_ptr, # Linear term in objective 
                          double *exp_ptr,          # stores exp(eta) 
                          double *expZ_accum_ptr,   # inner accumulation vector 
                          double *case_weight_ptr,  # case weights 
                          size_t *censoring_ptr,      # censoring indicator 
                          size_t *ordering_ptr,       # 0-based ordering of times 
                          size_t *rankmin_ptr,        # 0-based ranking with min tie breaking 
                          size_t ncase)               # how many subjects / times 

    void _update_outer_1st(double *linear_pred_ptr,     # Linear term in objective 
                           double *exp_accum_ptr,       # inner accumulation vector 
                           double *outer_accum_1st_ptr, # outer accumulation vector 
                           double *case_weight_ptr,     # case weights 
                           size_t *censoring_ptr,         # censoring indicator 
                           size_t *ordering_ptr,          # 0-based ordering of times 
                           size_t *rankmin_ptr,           # 0-based ranking with min tie breaking 
                           size_t ncase)                  # how many subjects / times 

    void _update_outer_2nd(double *linear_pred_ptr,     # Linear term in objective 
                           double *exp_accum_ptr,       # inner accumulation vector e^{\eta} 
                           double *expZ_accum_ptr,      # inner accumulation vector Ze^{\eta} 
                           double *outer_accum_2nd_ptr, # outer accumulation vector 
                           double *case_weight_ptr,     # case weights 
                           size_t *censoring_ptr,         # censoring indicator 
                           size_t *ordering_ptr,          # 0-based ordering of times 
                           size_t *rankmin_ptr,           # 0-based ranking with min tie breaking 
                           size_t ncase)                  # how many subjects / times 

    double _cox_objective(double *linear_pred_ptr,     # Linear term in objective 
                          double *inner_accum_ptr,     # inner accumulation vector 
                          double *outer_accum_1st_ptr, # outer accumulation vector 
                          double *case_weight_ptr,     # case weights 
                          size_t *censoring_ptr,         # censoring indicator 
                          size_t *ordering_ptr,          # 0-based ordering of times 
                          size_t *rankmin_ptr,           # 0-based ranking with min tie breaking 
                          size_t *rankmax_ptr,           # 0-based ranking with max tie breaking 
                          size_t ncase)                  # how many subjects / times 

    void _cox_gradient(double *gradient_ptr,        # Where gradient is stored 
                       double *exp_ptr,             # stores exp(eta) 
                       double *outer_accum_1st_ptr, # outer accumulation vector 
                       double *case_weight_ptr,     # case weights 
                       size_t *censoring_ptr,         # censoring indicator 
                       size_t *ordering_ptr,          # 0-based ordering of times 
                       size_t *rankmin_ptr,           # 0-based ranking with min tie breaking 
                       size_t *rankmax_ptr,           # 0-based ranking with max tie breaking 
                       size_t ncase)                  # how many subjects / times 

    void _cox_hessian(double *hessian_ptr,          # Where hessian is stored 
                      double *exp_ptr,              # stores exp(eta) 
                      double *right_vector_ptr,     # Right vector in Hessian
                      double *outer_accum_1st_ptr,  # outer accumulation vector used in outer prod "mean"
                      double *outer_accum_2nd_ptr,  # outer accumulation vector used in "2nd" moment
                      double *case_weight_ptr,      # case weights 
                      size_t *censoring_ptr,          # censoring indicator 
                      size_t *ordering_ptr,           # 0-based ordering of times 
                      size_t *rankmax_ptr,            # 0-based ranking with max tie breaking 
                      size_t ncase)                   # how many subjects / times 
   
def cox_objective(cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                  cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                  cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                  cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                  cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                  cnp.ndarray[DTYPE_intp_t, ndim=1] censoring,
                  cnp.ndarray[DTYPE_intp_t, ndim=1] ordering,
                  cnp.ndarray[DTYPE_intp_t, ndim=1] rankmin,
                  cnp.ndarray[DTYPE_intp_t, ndim=1] rankmax,
                  size_t ncase):
    """
    Compute the Cox negative log partial likelihood.

    The work buffers `exp_buffer`, `exp_accum` and `outer_1st_accum`
    are updated in place.

    Parameters
    ----------
    linear_pred : ndarray
        Linear predictor (natural parameter).
    exp_buffer : ndarray
        Buffer to store the case-weighted ``exp(linear_pred)``.
    exp_accum : ndarray
        Buffer for the reversed cumulative sums of `exp_buffer`
        over the risk sets.
    outer_1st_accum : ndarray
        Buffer for the outer accumulation used in the log term.
    case_weight : ndarray
        Non-negative case weights.
    censoring : ndarray
        Censoring indicator (1 for a failure, 0 for censored).
    ordering : ndarray
        0-based ordering of the event times, i.e. ``argsort(times)``.
    rankmin : ndarray
        0-based ranks of the event times with min tie breaking.
    rankmax : ndarray
        0-based ranks of the event times with max tie breaking.
    ncase : int
        Number of cases.

    Returns
    -------
    value : float
        The negative log partial likelihood.
    """

    # check shapes are correct

    assert(linear_pred.shape[0] == exp_buffer.shape[0])
    assert(linear_pred.shape[0] == exp_accum.shape[0])
    assert(linear_pred.shape[0] == outer_1st_accum.shape[0])
    assert(linear_pred.shape[0] == case_weight.shape[0])
    assert(linear_pred.shape[0] == censoring.shape[0])
    assert(linear_pred.shape[0] == ordering.shape[0])
    assert(linear_pred.shape[0] == rankmin.shape[0])
    assert(linear_pred.shape[0] == rankmax.shape[0])

    # ensure arrays are contiguous
    # may force a copy, but C code assumes contiguous
    # could be remedied by a stride argument

    linear_pred = np.ascontiguousarray(linear_pred)
    exp_buffer = np.ascontiguousarray(exp_buffer)
    exp_accum = np.ascontiguousarray(exp_accum)
    outer_1st_accum = np.ascontiguousarray(outer_1st_accum)
    case_weight = np.ascontiguousarray(case_weight)
    censoring = np.ascontiguousarray(censoring)
    ordering = np.ascontiguousarray(ordering)
    rankmin = np.ascontiguousarray(rankmin)
    rankmax = np.ascontiguousarray(rankmax)

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <size_t *>censoring.data,
                    <size_t *>ordering.data,
                    <size_t *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <size_t *>censoring.data,
                      <size_t *>ordering.data,
                      <size_t *>rankmin.data,
                      ncase)

    return _cox_objective(<double *>linear_pred.data,
                          <double *>exp_accum.data,
                          <double *>outer_1st_accum.data,
                          <double *>case_weight.data,
                          <size_t *>censoring.data,
                          <size_t *>ordering.data,
                          <size_t *>rankmin.data,
                          <size_t *>rankmax.data,
                          ncase)

def cox_gradient(cnp.ndarray[DTYPE_float_t, ndim=1] gradient,
                 cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                 cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                 cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                 cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                 cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                 cnp.ndarray[DTYPE_intp_t, ndim=1] censoring,
                 cnp.ndarray[DTYPE_intp_t, ndim=1] ordering,
                 cnp.ndarray[DTYPE_intp_t, ndim=1] rankmin,
                 cnp.ndarray[DTYPE_intp_t, ndim=1] rankmax,
                 size_t ncase):
    """
    Compute the gradient of the Cox negative log partial likelihood.

    The gradient is written into `gradient` (if it is contiguous) and
    the work buffers are updated in place. The remaining arguments are
    as in `cox_objective`.

    Returns
    -------
    gradient : ndarray
        Gradient with respect to `linear_pred`.
    """

    # check shapes are correct

    assert(gradient.shape[0] == linear_pred.shape[0])
    assert(gradient.shape[0] == exp_buffer.shape[0])
    assert(gradient.shape[0] == exp_accum.shape[0])
    assert(gradient.shape[0] == outer_1st_accum.shape[0])
    assert(gradient.shape[0] == case_weight.shape[0])
    assert(gradient.shape[0] == censoring.shape[0])
    assert(gradient.shape[0] == ordering.shape[0])
    assert(gradient.shape[0] == rankmin.shape[0])
    assert(gradient.shape[0] == rankmax.shape[0])

    # ensure arrays are contiguous
    # may force a copy, but C code assumes contiguous
    # could be remedied by a stride argument

    gradient = np.ascontiguousarray(gradient)
    linear_pred = np.ascontiguousarray(linear_pred)
    exp_buffer = np.ascontiguousarray(exp_buffer)
    exp_accum = np.ascontiguousarray(exp_accum)
    outer_1st_accum = np.ascontiguousarray(outer_1st_accum)
    case_weight = np.ascontiguousarray(case_weight)
    censoring = np.ascontiguousarray(censoring)
    ordering = np.ascontiguousarray(ordering)
    rankmin = np.ascontiguousarray(rankmin)
    rankmax = np.ascontiguousarray(rankmax)

    # this computes e^{\eta} and stores cumsum at rankmin

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <size_t *>censoring.data,
                    <size_t *>ordering.data,
                    <size_t *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <size_t *>censoring.data,
                      <size_t *>ordering.data,
                      <size_t *>rankmin.data,
                      ncase)

    _cox_gradient(<double *>gradient.data,
                  <double *>exp_buffer.data,
                  <double *>outer_1st_accum.data,
                  <double *>case_weight.data,
                  <size_t *>censoring.data,
                  <size_t *>ordering.data,
                  <size_t *>rankmin.data,
                  <size_t *>rankmax.data,
                  ncase)
    
    return gradient

def cox_hessian(cnp.ndarray[DTYPE_float_t, ndim=1] hessian,
                cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                cnp.ndarray[DTYPE_float_t, ndim=1] right_vector,
                cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] expZ_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] outer_2nd_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                cnp.ndarray[DTYPE_intp_t, ndim=1] censoring,
                cnp.ndarray[DTYPE_intp_t, ndim=1] ordering,
                cnp.ndarray[DTYPE_intp_t, ndim=1] rankmin,
                cnp.ndarray[DTYPE_intp_t, ndim=1] rankmax,
                size_t ncase):
    """
    Multiply the Hessian of the Cox loss by a vector.

    Computes the product of the Hessian of the Cox negative log partial
    likelihood with `right_vector`. The product is written into
    `hessian` (if it is contiguous) and the work buffers `exp_buffer`,
    `exp_accum`, `expZ_accum`, `outer_1st_accum` and `outer_2nd_accum`
    are updated in place. The remaining arguments are as in
    `cox_objective`.

    Parameters
    ----------
    hessian : ndarray
        Buffer for the result.
    linear_pred : ndarray
        Linear predictor at which the Hessian is evaluated.
    right_vector : ndarray
        Vector multiplied on the right of the Hessian.

    Returns
    -------
    hessian : ndarray
        The Hessian at `linear_pred` times `right_vector`.
    """

    # ensure arrays are contiguous
    # may force a copy, but C code assumes contiguous
    # could be remedied by a stride argument

    hessian = np.ascontiguousarray(hessian)
    linear_pred = np.ascontiguousarray(linear_pred)
    right_vector = np.ascontiguousarray(right_vector)
    exp_buffer = np.ascontiguousarray(exp_buffer)
    exp_accum = np.ascontiguousarray(exp_accum)
    expZ_accum = np.ascontiguousarray(expZ_accum)
    outer_1st_accum = np.ascontiguousarray(outer_1st_accum)
    outer_2nd_accum = np.ascontiguousarray(outer_2nd_accum)
    case_weight = np.ascontiguousarray(case_weight)
    censoring = np.ascontiguousarray(censoring)
    ordering = np.ascontiguousarray(ordering)
    rankmin = np.ascontiguousarray(rankmin)
    rankmax = np.ascontiguousarray(rankmax)

    # check shapes are correct

    assert(hessian.shape[0] == linear_pred.shape[0])
    assert(hessian.shape[0] == right_vector.shape[0])
    assert(hessian.shape[0] == exp_buffer.shape[0])
    assert(hessian.shape[0] == exp_accum.shape[0])
    assert(hessian.shape[0] == expZ_accum.shape[0])
    assert(hessian.shape[0] == outer_1st_accum.shape[0])
    assert(hessian.shape[0] == outer_2nd_accum.shape[0])
    assert(hessian.shape[0] == case_weight.shape[0])
    assert(hessian.shape[0] == censoring.shape[0])
    assert(hessian.shape[0] == ordering.shape[0])
    assert(hessian.shape[0] == rankmin.shape[0])
    assert(hessian.shape[0] == rankmax.shape[0])

    # this computes e^{\eta} and stores its reversed cumsum in exp_accum

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <size_t *>censoring.data,
                    <size_t *>ordering.data,
                    <size_t *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <size_t *>censoring.data,
                      <size_t *>ordering.data,
                      <size_t *>rankmin.data,
                      ncase)

    _update_cox_expZ(<double *>linear_pred.data,
                     <double *>right_vector.data,
                     <double *>exp_buffer.data,
                     <double *>expZ_accum.data,
                     <double *>case_weight.data,
                     <size_t *>censoring.data,
                     <size_t *>ordering.data,
                     <size_t *>rankmin.data,
                     ncase)

    _update_outer_2nd(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>expZ_accum.data,
                      <double *>outer_2nd_accum.data,
                      <double *>case_weight.data,
                      <size_t *>censoring.data,
                      <size_t *>ordering.data,
                      <size_t *>rankmin.data,
                      ncase)

    _cox_hessian(<double *>hessian.data,
                 <double *>exp_buffer.data,
                 <double *>right_vector.data,
                 <double *>outer_1st_accum.data,
                 <double *>outer_2nd_accum.data,
                 <double *>case_weight.data,
                 <size_t *>censoring.data,
                 <size_t *>ordering.data,
                 <size_t *>rankmax.data,
                 ncase)
    
    return hessian
              
