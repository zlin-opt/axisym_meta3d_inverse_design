#ifndef GUARD_vec_h
#define GUARD_vec_h

#include <assert.h>
#include "petsc.h"
#include "array2vec.h"
#include "type.h"

void setlayer_eps(Vec eps,
		  PetscInt Nr, PetscInt Nz,
		  PetscInt num_layers, PetscInt *zstarts, PetscInt *thickness,
		  PetscScalar *epsilon);

void vecfill_zslice(Vec v, PetscScalar *vr, PetscScalar *vtheta, PetscInt Nr, PetscInt Nz, PetscInt iz);

#endif
