#ifndef GUARD_ovmat_h
#define GUARD_ovmat_h

#include "petsc.h"
#include "type.h"

void create_ovmat(MPI_Comm comm, Mat *Wout, PetscInt *mrows_per_cell, PetscInt *cell_start, PetscInt nr, PetscInt pr, PetscInt ncells, PetscInt nlayers, PetscInt pmlr0, PetscInt pmlr1, PetscInt *mz, PetscInt mzslab);

#endif
