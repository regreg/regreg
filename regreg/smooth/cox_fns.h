#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

void _update_cox_exp(double *linear_pred_ptr, /* Linear term in objective */
		     double *exp_ptr,          /* stores exp(eta) */
		     double *exp_accum_ptr,   /* inner accumulation vector */
		     double *case_weight_ptr, /* case weights */
		     size_t *censoring_ptr,     /* censoring indicator */
		     size_t *ordering_ptr,      /* 0-based ordering of times */
		     size_t *rankmin_ptr,       /* 0-based ranking with min tie breaking */
		     size_t ncase               /* how many subjects / times */
		     );       

void _update_cox_expZ(double *linear_pred_ptr,  /* Linear term in objective */
		      double *right_vector_ptr, /* Linear term in objective */
		      double *exp_ptr,          /* stores exp(eta) */
		      double *expZ_accum_ptr,   /* inner accumulation vector */
		      double *case_weight_ptr,  /* case weights */
		      size_t *censoring_ptr,      /* censoring indicator */
		      size_t *ordering_ptr,       /* 0-based ordering of times */
		      size_t *rankmin_ptr,        /* 0-based ranking with min tie breaking */
		      size_t ncase                /* how many subjects / times */
		      );       

void _update_outer_1st(double *linear_pred_ptr,     /* Linear term in objective */
		       double *exp_accum_ptr,       /* inner accumulation vector */
		       double *outer_accum_1st_ptr, /* outer accumulation vector */
		       double *case_weight_ptr,     /* case weights */
		       size_t *censoring_ptr,         /* censoring indicator */
		       size_t *ordering_ptr,          /* 0-based ordering of times */
		       size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		       size_t ncase                   /* how many subjects / times */
		       );       

void _update_outer_2nd(double *linear_pred_ptr,     /* Linear term in objective */
		       double *exp_accum_ptr,       /* inner accumulation vector e^{\eta} */
		       double *expZ_accum_ptr,      /* inner accumulation vector  Ze^{\eta} */
		       double *outer_accum_2nd_ptr, /* outer accumulation vector */
		       double *case_weight_ptr,     /* case weights */
		       size_t *censoring_ptr,         /* censoring indicator */
		       size_t *ordering_ptr,          /* 0-based ordering of times */
		       size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		       size_t ncase                   /* how many subjects / times */
		       );

double _cox_objective(double *linear_pred_ptr,     /* Linear term in objective */
		      double *inner_accum_ptr,     /* inner accumulation vector */
		      double *outer_accum_1st_ptr, /* outer accumulation vector */
		      double *case_weight_ptr,     /* case weights */
		      size_t *censoring_ptr,         /* censoring indicator */
		      size_t *ordering_ptr,          /* 0-based ordering of times */
		      size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		      size_t *rankmax_ptr,           /* 0-based ranking with max tie breaking */
		      size_t ncase                   /* how many subjects / times */
		      );       

void _cox_gradient(double *gradient_ptr,        /* Where gradient is stored */
		   double *exp_ptr,             /* stores exp(eta) */
		   double *outer_accum_1st_ptr, /* outer accumulation vector */
		   double *case_weight_ptr,     /* case weights */
		   size_t *censoring_ptr,         /* censoring indicator */
		   size_t *ordering_ptr,          /* 0-based ordering of times */
		   size_t *rankmin_ptr,           /* 0-based ranking with min tie breaking */
		   size_t *rankmax_ptr,           /* 0-based ranking with max tie breaking */
		   size_t ncase                   /* how many subjects / times */
		   );

void _cox_hessian(double *hessian_ptr,          /* Where hessian is stored */
		  double *exp_ptr,              /* stores exp(eta) */
		  double *right_vector_ptr,     /* Right vector in Hessian */
		  double *outer_accum_1st_ptr,  /* outer accumulation vector used in outer prod "mean"*/
		  double *outer_accum_2nd_ptr,  /* outer accumulation vector used in "2nd" moment*/
		  double *case_weight_ptr,     /* case weights */
		  size_t *censoring_ptr,          /* censoring indicator */
		  size_t *ordering_ptr,           /* 0-based ordering of times */
		  size_t *rankmax_ptr,            /* 0-based ranking with max tie breaking */
		  size_t ncase                    /* how many subjects / times */
		  );


#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */
