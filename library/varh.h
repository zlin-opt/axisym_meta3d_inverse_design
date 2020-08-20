#ifndef GUARD_varh_h
#define GUARD_varh_h

#include "petsc.h"
#include "type.h"

void varh_expand(PetscScalar *hdof, PetscScalar *edof, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed);

void varh_contract(PetscScalar *egrad, PetscScalar *hgrad, PetscScalar *hdof, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed);

void density_filter(MPI_Comm comm, Mat *Qout, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscReal fr, PetscReal sigma, PetscInt normalized);
//weight = exp(-dist^2/sigma^2) 

#endif
