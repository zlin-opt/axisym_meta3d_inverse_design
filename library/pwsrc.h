#ifndef GUARD_pwsrc_h
#define GUARD_pwsrc_h

#include "petsc.h"
#include "ovmat.h"
#include "array2vec.h"
#include "misc.h"
#include "type.h"

void Jm(PetscScalar *Jmr, PetscScalar *Jmt,
	PetscInt ir_start, PetscInt nr_segment, PetscReal dr, PetscReal dz,
	PetscInt m,
	PetscReal k, PetscReal alpha, PetscReal phi,
	PetscScalar Ax, PetscScalar Ay);

void m_select(PetscInt num_m_max, PetscReal r_min, PetscReal r_max, PetscReal k, PetscReal phi, PetscReal alpha, PetscReal cutoff,  PetscInt *num_m, PetscInt **mlist);

void make_sigmahat(Vec sigmahat, PetscInt Nr, PetscInt Nz, PetscInt pol);

void m_select_nonneg(PetscInt num_m_max, PetscReal r_min, PetscReal r_max, PetscReal k, PetscReal phi, PetscReal alpha, PetscReal cutoff,  PetscInt *num_m, PetscInt **mlist);

#endif
