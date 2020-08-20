#include "vec.h"

#undef __FUNCT__
#define __FUNCT__ "setlayer_eps"
void setlayer_eps(Vec eps,
		  PetscInt Nr, PetscInt Nz,
		  PetscInt num_layers, PetscInt *zstarts, PetscInt *thickness,
		  PetscScalar *epsilon)
{

  PetscInt Nc = 3;
  PetscScalar *_eps = (PetscScalar *)malloc(Nr*Nz*Nc*sizeof(PetscScalar));
  
  for(PetscInt ic=0;ic<Nc;ic++){
    for(PetscInt iz=0;iz<Nz;iz++){
      for(PetscInt ir=0;ir<Nr;ir++){
	PetscInt i = ir+Nr*iz+Nr*Nz*ic;
	_eps[i]=0;
	for(PetscInt j=0;j<num_layers;j++){
	  if(iz>=zstarts[j] && iz<zstarts[j]+thickness[j])
	    _eps[i]=epsilon[j];
	}
      }
    }
  }

  array2mpi(_eps,SCAL,eps);

  free(_eps);
  
}

#undef __FUNCT__
#define __FUNCT__ "vecfill_zslice"
void vecfill_zslice(Vec v, PetscScalar *vr, PetscScalar *vtheta, PetscInt Nr, PetscInt Nz, PetscInt iz)
{

  PetscInt ns,ne;
  VecGetOwnershipRange(v,&ns,&ne);
  for( PetscInt j=ns;j<ne;j++)
    {
      PetscInt k;
      PetscInt jr=(k=j)%Nr;
      PetscInt jz=(k/=Nr)%Nz;
      PetscInt jc=(k/=Nz)%3;
      if(jz==iz && jc==0)
	VecSetValue(v,j, vr[jr],     INSERT_VALUES);
      if(jz==iz && jc==1)
	VecSetValue(v,j, vtheta[jr], INSERT_VALUES);
    }
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

}
