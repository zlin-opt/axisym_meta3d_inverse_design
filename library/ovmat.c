#include "ovmat.h"

// rows are indexed ir,iz,ilayer,icell from fastest to slowest index
// columns are "globally" indexed jjr,jz,jlayer where jjr spans all the cells jjr = jr + nr*jcell

#undef __FUNCT__
#define __FUNCT__ "create_ovmat"
void create_ovmat(MPI_Comm comm, Mat *Wout, PetscInt *mrows_per_cell, PetscInt *cell_start, PetscInt nr, PetscInt pr, PetscInt ncells, PetscInt nlayers, PetscInt pmlr0, PetscInt pmlr1, PetscInt *mz, PetscInt mzslab)
{

  //PetscPrintf(comm,"Creating the overlap extension matrix.\n");

  PetscInt mr0=   nr+pr;
  PetscInt mr =pr+nr+pr;

  PetscInt mrows=0;
  for(PetscInt j=0;j<ncells;j++){
    mrows_per_cell[j] = 0;
    for( PetscInt i=0;i<nlayers;i++){
      if(j==0)
	mrows_per_cell[j] += (mzslab==0) ? mr0*mz[i] : mr0 ;
      else
	mrows_per_cell[j] += (mzslab==0) ? mr *mz[i] : mr  ;
    }
    cell_start[j]=mrows;
    mrows += mrows_per_cell[j];
  }

  PetscInt idset_ir[mrows],idset_iz[mrows],idset_il[mrows],idset_ic[mrows];
  PetscInt id=0;
  for(PetscInt ic=0;ic<ncells;ic++){
    PetscInt tmpmr = (ic==0) ? mr0 : mr;
    for(PetscInt il=0;il<nlayers;il++){
      PetscInt tmpmz = (mzslab==0) ? mz[il] : 1;
      for(PetscInt iz=0;iz<tmpmz;iz++){
	for(PetscInt ir=0;ir<tmpmr;ir++){
	  idset_ir[id]=ir;
	  idset_iz[id]=iz;
	  idset_il[id]=il;
	  idset_ic[id]=ic;
	  id++;
	}
      }
    }
  }
  
  PetscInt Nr=nr*ncells;
  PetscInt ncols = 0;
  PetscInt nrnz[nlayers];
  for(PetscInt i=0;i<nlayers;i++){
    nrnz[i]= ncols;
    ncols += (mzslab==0) ? Nr*mz[i] : Nr ;
  }

  Mat W;

  PetscInt ns,ne;
  PetscInt i,j;
  PetscInt ir,iz,il,ic;
  PetscInt jr,jz,jl,jc,jjr;
  PetscScalar val;

  MatCreate(comm,&W);
  MatSetType(W,MATMPIAIJ);
  MatSetSizes(W,PETSC_DECIDE,PETSC_DECIDE, mrows,ncols);
  MatMPIAIJSetPreallocation(W, 1, PETSC_NULL, 1, PETSC_NULL);

  MatGetOwnershipRange(W, &ns, &ne);

  for(i=ns;i<ne;i++){

    val=1.0;

    ir=idset_ir[i];
    iz=idset_iz[i];
    il=idset_il[i];
    ic=idset_ic[i];
    
    if(ic>0 && ir < pr){

      jc = ic-1;
      jr = nr - (pr-ir); //jr = (nr - 1) - (pr - 1 -ir);

    }else if(ic>0 && ir >= pr+nr){

      jc = ic+1;
      jr = ir - (pr+nr);

    }else if(ic>0 && ir>=pr && ir<pr+nr){

      jc = ic;
      jr = ir - pr;

    }else if(ic==0 && ir >= nr){

      jc = ic+1;
      jr = ir - nr;

    }else{

      jc = ic;
      jr = ir;

    }

    if(jc>=ncells){

      jc  = ncells-1;
      val = 0.0;

    }

    if( ic>0 && (ir<pmlr0 || ir>=mr-pmlr1) )
      val=0.0;
    if(ic==0 && ir>mr0-pmlr1)
      val=0.0;

    jl=il;
    jz=iz;
    jjr=jr+nr*jc;
    j=jjr+Nr*jz+nrnz[jl];

    MatSetValue(W,i,j,val,ADD_VALUES);

  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  *Wout = W;

}
