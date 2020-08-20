#ifndef GUARD_qforms_h
#define GUARD_qforms_h

#include "petsc.h"
#include "type.h"
#include "maxwell.h"
#include "greendyadic.h"
#include "vec.h"

void create_Qp(MPI_Comm comm, Mat *Qp_out,
	       PetscInt nr, PetscInt nz,
	       PetscInt npmlr0, PetscInt npmlr1, PetscInt npmlz0, PetscInt npmlz1,
	       PetscReal dr, PetscReal dz,
	       PetscReal omega,
	       PetscInt m,
	       PetscInt ir_lstart, PetscInt nr_mtr, PetscInt ir_gstart, PetscInt iz_mtr);

void gforms(MPI_Comm comm, Vec gex, Vec gey, Vec ghx, Vec ghy,
	    PetscInt nr, PetscInt nz,
	    PetscInt npmlr0, PetscInt npmlr1, PetscInt npmlz0, PetscInt npmlz1,
	    PetscReal dr, PetscReal dz,
	    PetscReal omega,
	    PetscInt m,
	    PetscInt ir_lstart, PetscInt nr_mtr, PetscInt ir_gstart, PetscInt iz_mtr,
	    PetscReal xfar, PetscReal yfar, PetscReal zfar, PetscReal znear,
	    PetscReal mu, PetscReal eps);

#endif
