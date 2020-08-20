#ifndef GUARD_dof2dom_h
#define GUARD_dof2dom_h

#include <assert.h>
#include "petsc.h"
#include "type.h"

void create_Ainterp(MPI_Comm comm, Mat *A_out,
		    PetscInt Nr, PetscInt Nz,
		    PetscInt Mro, PetscInt Mr, PetscInt *Mzo, PetscInt *Mz, PetscInt Mzslab,
		    PetscInt nlayers);

/*
Create the matrix that extends a DOF array to the simulation domain.
The simulation domain consists of multiple "vertical" layers specified by 
    nlayers, 
    Mzo[nlayers] (starting z-coordinates in pixels)
    Mz[nlayers] (layers' thicknesses in pixels)
    Mzslab ( 0 dielectric varies along z within each layer; 1 invariant along z within each layer)
Note that the (flattened) size of the simulation domain = Nr*Nz*3 = #rows.
Within Nr, the DOFs are filled starting from ir=Mro to ir<Mro+Mr. 
Usually set Mro = 0 and Mr = Nr.
The size of the DOF array should be 
    Mr*nlayers if Mzslab = 1 
OR  Mr*sum(Mz) if Mzslab = 0
Note that the matrix is created under the MPI subcommunicator "comm" which can be different from PETSC_COMM_WORLD.
*/

#endif
