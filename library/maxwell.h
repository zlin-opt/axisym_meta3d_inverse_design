#ifndef GUARD_maxwell_h
#define GUARD_maxwell_h

#include <assert.h>
#include "petsc.h"
#include "type.h"

PetscScalar pml_s(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega);
PetscScalar pml_p(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega);

void create_De(MPI_Comm comm, Mat *De_out,
	       PetscInt Nr, PetscInt Nz,
	       PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
	       PetscReal dr, PetscReal dz, PetscReal R0,
	       PetscReal omega,
	       PetscInt m);

void create_Dh(MPI_Comm comm, Mat *Dh_out,
	       PetscInt Nr, PetscInt Nz,
	       PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
	       PetscReal dr, PetscReal dz, PetscReal R0,
	       PetscReal omega,
	       PetscInt m);

void create_DDe(MPI_Comm comm, Mat *DDe_out,
		PetscInt Nr, PetscInt Nz,
		PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dr, PetscReal dz, PetscReal R0,
		PetscReal omega,
		PetscInt m);

void syncE(MPI_Comm comm, Mat *Ae_out, PetscInt Nr, PetscInt Nz);
void syncH(MPI_Comm comm, Mat *Ah_out, PetscInt Nr, PetscInt Nz);

#endif
