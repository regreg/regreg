#include <stddef.h>
#include <math.h> // for exp, log

// This provides (as a function of linear predictor), the Cox partial likelihood, 
// its gradient and its Hessian acting on an n \times k matrix on the right.

// This forms the vector of weights
// W_i = \sum_{j: T_j \geq T_{(i)}} \exp(\eta_j)
// so it is ordered according to argsort(T)

void _update_cox_exp(double *linear_pred_ptr, /* Linear term in objective */
		     double *exp_ptr,         /* stores exp(eta) */
		     double *exp_accum_ptr,   /* inner accumulation vector */
		     double *case_weight_ptr, /* case weights */
		     size_t *censoring_ptr,     /* censoring indicator */
		     size_t *ordering_ptr,      /* 0-based ordering of times */
		     size_t *rankmin_ptr,       /* 0-based ranking with min tie breaking */
		     size_t ncase               /* how many subjects / times */
		     )       
{
  size_t idx;
  size_t order_idx;
  double linear_pred, case_weight;
  double *exp_accum;
  double *exp_val_ptr;
  double cur_val = 0;

  // reversed reverse cumsum of exp(eta)

  for (idx=0; idx<ncase; idx++) {
    order_idx = *((size_t *) ordering_ptr + (ncase - 1 - idx));
    linear_pred = *((double *) linear_pred_ptr + order_idx);
    case_weight = *((double *) case_weight_ptr + order_idx);
    exp_val_ptr = ((double *) exp_ptr + order_idx);
    *exp_val_ptr = exp(linear_pred) * case_weight;
    cur_val = cur_val + (*exp_val_ptr);
    exp_accum = ((double *) exp_accum_ptr + (ncase - 1 - idx));
    *exp_accum = cur_val;
  }

}

void _update_cox_expZ(double *linear_pred_ptr,  /* Linear term in objective */
		      double *right_vector_ptr, /* Linear term in objective */
		      double *exp_ptr,          /* stores exp(eta) */
		      double *expZ_accum_ptr,   /* inner accumulation vector */
		      double *case_weight_ptr,  /* case weights */
		      size_t *censoring_ptr,      /* censoring indicator */
		      size_t *ordering_ptr,       /* 0-based ordering of times */
		      size_t *rankmin_ptr,        /* 0-based ranking with min tie breaking */
		      size_t ncase                /* how many subjects / times */
		      )       
{
  size_t idx;
  size_t order_idx;
  double right_vector, exp_val;
  double *expZ_accum;
  double cur_val = 0;

  // reversed reverse cumsum of exp(eta)

  for (idx=0; idx<ncase; idx++) {
    order_idx = *((size_t *) ordering_ptr + (ncase - 1 - idx));
    exp_val = *((double *) exp_ptr + order_idx);
    right_vector = *((double *) right_vector_ptr + order_idx);
    cur_val = cur_val + right_vector * exp_val;
    expZ_accum = ((double *) expZ_accum_ptr + (ncase - 1 - idx));
    *expZ_accum = cur_val;
  }

}

void _update_outer_1st(double *linear_pred_ptr,     /* Linear term in objective */
		       double *exp_accum_ptr,       /* inner accumulation vector */
		       double *outer_1st_accum_ptr, /* outer accumulation vector */
		       double *case_weight_ptr,     /* case weights */
		       size_t *censoring_ptr,         /* censoring indicator */
		       size_t *ordering_ptr,          /* 0-based ordering of times */
		       size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		       size_t ncase                   /* how many subjects / times */
		       )       
{
  size_t idx;
  size_t order_idx, rankmin_idx;
  double case_weight;
  size_t censoring;
  double *exp_accum, *outer_1st_accum;
  double cur_val = 0;

  // accumulate inverse cumsums at rankmin
  // i-th value is \sum_{j=1}^i 1 / W(r[o[j]])  where r is rankmin of times so r[o]
  // is rankmin of ordered times

  cur_val = 0;
  for (idx=0; idx<ncase; idx++) {
    order_idx = *((size_t *) ordering_ptr + idx);
    rankmin_idx = *((size_t *) rankmin_ptr + order_idx);
    exp_accum = ((double *) exp_accum_ptr + rankmin_idx);
    censoring = *((size_t *) censoring_ptr + order_idx);
    case_weight = *((double *) case_weight_ptr + order_idx);
    cur_val = cur_val + censoring * case_weight / *exp_accum;
    outer_1st_accum = ((double *) outer_1st_accum_ptr + idx);
    *outer_1st_accum = cur_val;
  }

}

void _update_outer_2nd(double *linear_pred_ptr,     /* Linear term in objective */
		       double *exp_accum_ptr,       /* inner accumulation vector e^{\eta} */
		       double *expZ_accum_ptr,      /* inner accumulation vector  Ze^{\eta} */
		       double *outer_2nd_accum_ptr, /* outer accumulation vector */
		       double *case_weight_ptr,     /* case weights */
		       size_t *censoring_ptr,         /* censoring indicator */
		       size_t *ordering_ptr,          /* 0-based ordering of times */
		       size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		       size_t ncase                   /* how many subjects / times */
		       )       
{
  size_t idx;
  size_t order_idx, rankmin_idx;
  double case_weight;
  size_t censoring;
  double *expZ_accum, *exp_accum, *outer_2nd_accum;
  double cur_val = 0;

  // accumulate inverse cumsums at rankmin
  // i-th value is \sum_{j=1}^i 1 / W(r[o[j]])  where r is rankmin of times so r[o]
  // is rankmin of ordered times

  for (idx=0; idx<ncase; idx++) {
    order_idx = *((size_t *) ordering_ptr + idx);
    rankmin_idx = *((size_t *) rankmin_ptr + order_idx);
    expZ_accum = ((double *) expZ_accum_ptr + rankmin_idx);
    exp_accum = ((double *) exp_accum_ptr + rankmin_idx);
    censoring = *((size_t *) censoring_ptr + order_idx);
    case_weight = *((double *) case_weight_ptr + order_idx);
    cur_val = cur_val + censoring * case_weight * (*expZ_accum) / ((*exp_accum) * (*exp_accum));
    outer_2nd_accum = ((double *) outer_2nd_accum_ptr + idx);
    *outer_2nd_accum = cur_val;
  }

}

// Objective value

double _cox_objective(double *linear_pred_ptr,     /* Linear term in objective */
		      double *exp_accum_ptr,       /* accumulation of exp(eta) */
		      double *outer_1st_accum_ptr, /* outer accumulation vector */
		      double *case_weight_ptr,     /* case weights */
		      size_t *censoring_ptr,         /* censoring indicator */
		      size_t *ordering_ptr,          /* 0-based ordering of times */
		      size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		      size_t *rankmax_ptr,           /* 0-based ranking with max tie breaking */
		      size_t ncase                   /* how many subjects / times */
		      )       
{
  size_t idx, rankmin_idx;
  double linear_pred, exp_accum, case_weight;
  size_t censoring;
  double cur_val = 0;

  // ensure you have updated the inner / outer accumulation
  // vectors with current linear predictors
  // this can be done in the wrapper _update_cox_weights

  for (idx=0; idx<ncase; idx++) {
    rankmin_idx = *((size_t *) rankmin_ptr + idx);
    exp_accum = *((double *) exp_accum_ptr + rankmin_idx);
    censoring = *((size_t *) censoring_ptr + idx);
    case_weight = *((double *) case_weight_ptr + idx);
    linear_pred = *((double *) linear_pred_ptr + idx);
    cur_val += censoring * case_weight * (log(exp_accum) - linear_pred);
  }

  return(cur_val);

}

void _cox_gradient(double *gradient_ptr,        /* Where gradient is stored */
		   double *exp_ptr,             /* stores exp(eta) */
		   double *outer_1st_accum_ptr, /* outer accumulation vector */
		   double *case_weight_ptr,     /* case weights */
		   size_t *censoring_ptr,         /* censoring indicator */
		   size_t *ordering_ptr,          /* 0-based ordering of times */
		   size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		   size_t *rankmax_ptr,           /* 0-based ranking with max tie breaking */
		   size_t ncase                   /* how many subjects / times */
		   )
{
  size_t idx, rankmax_idx;
  double outer_1st_accum, exp_val, case_weight;
  double *gradient;
  size_t censoring;

  // ensure you have updated the inner / outer accumulation
  // vectors with current linear predictors
  // this can be done in the wrapper _update_cox_weights

  // fill in entries of gradient

  for (idx=0; idx<ncase; idx++) {
    censoring = *((size_t *) censoring_ptr + idx);
    case_weight = *((double *) case_weight_ptr + idx);
    rankmax_idx = *((size_t *) rankmax_ptr + idx);
    outer_1st_accum = *((double *) outer_1st_accum_ptr + rankmax_idx);
    exp_val = *((double *) exp_ptr + idx);
    gradient = ((double *) gradient_ptr + idx);
    *gradient = outer_1st_accum * exp_val - censoring * case_weight;
  }

}

void _cox_hessian(double *hessian_ptr,          /* Where hessian is stored */
		  double *exp_ptr,              /* stores exp(eta) */
		  double *right_vector_ptr,     /* Right vector in Hessian */
		  double *outer_1st_accum_ptr,  /* outer accumulation vector used in outer prod "mean"*/
		  double *outer_2nd_accum_ptr,  /* outer accumulation vector used in "2nd" moment*/
		  double *case_weight_ptr,      /* case weights */
		  size_t *censoring_ptr,          /* censoring indicator */
		  size_t *ordering_ptr,           /* 0-based ordering of times */
		  size_t *rankmax_ptr,            /* 0-based ranking with max tie breaking */
		  size_t ncase                    /* how many subjects / times */
		  )
{
  size_t idx, rankmax_idx;
  double outer_1st_accum, outer_2nd_accum, right_vector, exp_val;
  double *hessian;

  // ensure you have updated the inner / outer accumulation
  // vectors with current linear predictors
  // this can be done in the wrapper with _update_cox_hessian

  // fill in entries of hessian

  for (idx=0; idx<ncase; idx++) {
    rankmax_idx = *((size_t *) rankmax_ptr + idx);
    outer_1st_accum = *((double *) outer_1st_accum_ptr + rankmax_idx);
    outer_2nd_accum = *((double *) outer_2nd_accum_ptr + rankmax_idx);
    exp_val = *((double *) exp_ptr + idx);
    right_vector = *((double *) right_vector_ptr + idx);
    hessian = ((double *) hessian_ptr + idx);
    *hessian =  exp_val * (outer_1st_accum * right_vector - outer_2nd_accum);
  }

}

