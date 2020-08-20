#ifndef GUARD_greendyadic_h
#define GUARD_greendyadic_h

#include "petsc.h"
#include "cubature.h"

PetscInt fE(unsigned ndim, const PetscReal *t, void *data,
	    unsigned fdim, PetscReal *fval);

void near2far(Vec Er, Vec Et, Vec Hr, Vec Ht,
	      PetscReal x, PetscReal y, PetscReal z, PetscReal z0,
	      PetscInt nr_mtr, PetscReal dr,
	      PetscInt num_m, PetscInt *mlist,
	      PetscReal omega, PetscReal mu, PetscReal eps,
	      PetscScalar *farfield);

#endif
