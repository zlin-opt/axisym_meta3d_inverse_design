#ifndef GUARD_optitemp_h
#define GUARD_optitemp_h

#include "petsc.h"
#include "nlopt.h"
#include "output.h"
#include "type.h"

typedef struct{

  nlopt_algorithm outer;
  nlopt_algorithm inner;
  PetscInt maxeval;
  PetscInt maxtime;
  PetscInt maxobj;
  
} alg_;

PetscReal optimize_generic(PetscInt DegFree, PetscReal *epsopt,
			PetscReal *lb, PetscReal *ub,
			nlopt_func obj, void *objdata,
			nlopt_func *constraint, void **constrdata, PetscInt nconstraints,
			alg_ alg,
			nlopt_result *result);

PetscReal dummy_obj(PetscInt ndofAll_with_dummy, PetscReal *dofAll_with_dummy, PetscReal *dofgradAll_with_dummy, void *print_at);

#endif
