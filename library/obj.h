#ifndef GUARD_obj_h
#define GUARD_obj_h

#include "petsc.h"

typedef struct{

  //common params
  PetscInt nfreqs;
  PetscInt nangles;
  PetscInt *npols;
  PetscReal *freqs;
  PetscReal *angles;
  PetscInt **num_m;
  PetscInt ***mlist;
  PetscInt nspecs;
  PetscInt nsims;
  PetscInt nsolves;
  PetscInt **idset_specs;
  PetscInt **idset_sims;
  PetscInt ***idset_solves;
  
  PetscInt nr;
  PetscInt pr;
  PetscInt pmlr;
  PetscInt ncells;
  PetscInt nz;
  PetscInt pmlz;
  PetscReal dr;
  PetscReal dz;
  PetscInt iz_src;
  PetscInt iz_mtr;
  PetscInt nlayers_active;
  PetscInt *mz;
  PetscInt mzsum;  
  
  Mat W;
  PetscReal filter_beta;
  PetscInt zfixed;
  
  MPI_Comm subcomm;
  PetscInt nsolves_per_comm; 
  PetscInt ncomms;
  PetscInt colour;

  //distributed vecs and mats
  Vec *epsDiff;
  Vec *epsBkg;
  
  PetscScalar ***Jr;
  PetscScalar ***Jt;
  
  Mat *A;
  Mat *DDe;

  Mat *Qp;
  Mat *Qp_minus;
  Vec *gex;
  Vec *gey;
  Vec *ghx;
  Vec *ghy;
  Vec *gex_minus;
  Vec *gey_minus;
  Vec *ghx_minus;
  Vec *ghy_minus;
  Vec *sigmahat;
  
  KSP *ksp;
  PetscInt *its;
  PetscInt maxit;

  PetscReal *airyfactor;
  
} params_;

#endif
