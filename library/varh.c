#include "varh.h"

static PetscReal stepfunc(PetscReal dz, PetscReal eta, PetscReal beta){

  PetscReal b = (beta>1e-2) ? beta : 1e-2;
  
  PetscReal r1 = tanh(b*eta) + tanh(b*(dz -eta));
  PetscReal r2 = tanh(b*eta) + tanh(b*(1.0-eta));

  return 1.0-(r1/r2);

}

static PetscReal stepgrad(PetscReal dz, PetscReal eta, PetscReal beta){

  PetscReal b = (beta>1e-2) ? beta : 1e-2;
  
  PetscReal csch = 1.0/sinh(b);
  PetscReal sech = 1.0/cosh(b*(dz-eta));

  return b*csch*sech*sech*sinh(b*dz)*sinh(b-b*dz);

}

static PetscReal piecewise(PetscReal rho, PetscReal beta){

  PetscReal eta=0.5;
  PetscReal rout;

  if(beta<1e-3){

    rout=rho;

  }else{

    if(rho>=0 && rho<=eta)
      rout=eta * ( exp(-beta*(1-rho/eta)) - (1-rho/eta)*exp(-beta) );
    else if(rho>eta && rho<=1.0)
      rout=(1-eta) * ( 1 - exp(-beta*(rho-eta)/(1-eta)) + (rho-eta)*exp(-beta)/(1-eta) ) + eta;
    else if(rho<0)
      rout=0;
    else
      rout=1;

  }

  return rout;

}

static PetscReal piecewisegrad(PetscReal rho, PetscReal beta){

  PetscReal eta=0.5;
  PetscReal rg;

  if(beta<1e-3){

    rg =1.0;

  }else{

    if(rho>=0 && rho<=eta)
      rg =eta * ( (beta/eta)*exp(-beta*(1-rho/eta)) + exp(-beta)/eta );
    else if(rho>eta && rho<=1.0)
      rg =(1-eta) * ( beta/(1-eta) * exp(-beta*(rho-eta)/(1-eta)) + exp(-beta)/(1-eta) );
    else if(rho<0)
      rg =0;
    else
      rg =0;

  }

  return rg;

}

// the output indexing as ir,iz,ilayer from fastest to slowest index
// the input indexing as jx,jlayer 

#undef __FUNCT__
#define __FUNCT__ "varh_expand"
void varh_expand(PetscScalar *hdof, PetscScalar *edof, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed)
{

  PetscInt Nr=nr*ncells;
  PetscInt id=0;
  for (PetscInt il=0;il<nlayers;il++){
    for (PetscInt iz=0;iz<mz[il];iz++){
      for (PetscInt ir=0;ir<Nr;ir++){

	if(zfixed==0){
	  PetscReal dz=(PetscReal)iz/(PetscReal)mz[il];
	  edof[id]=stepfunc(dz,creal(hdof[ir+Nr*il]),beta) + PETSC_i*0.0;
	}else{
	  edof[id]=piecewise(creal(hdof[ir+Nr*il]),beta) + PETSC_i*0.0;
	}
	
	id++;
      }
    }
  }
    

}


#undef __FUNCT__
#define __FUNCT__ "varh_contract"
void varh_contract(PetscScalar *egrad, PetscScalar *hgrad, PetscScalar *hdof, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed)
{

  PetscInt Nr=nr*ncells;
  PetscInt id=0;
  for (PetscInt il=0;il<nlayers;il++){
    for (PetscInt iz=0;iz<mz[il];iz++){
      for (PetscInt ir=0;ir<Nr;ir++){

	if(zfixed==0){
	  PetscReal dz=(PetscReal)iz/(PetscReal)mz[il];
	  if(iz==0)
	    hgrad[ir+Nr*il]  = stepgrad(dz,creal(hdof[ir+Nr*il]),beta)*egrad[id];
	  else
	    hgrad[ir+Nr*il] += stepgrad(dz,creal(hdof[ir+Nr*il]),beta)*egrad[id];
	}else{
	  if(iz==0)
	    hgrad[ir+Nr*il]  = piecewisegrad(creal(hdof[ir+Nr*il]),beta)*egrad[id];
	  else
	    hgrad[ir+Nr*il] += piecewisegrad(creal(hdof[ir+Nr*il]),beta)*egrad[id];
	}
	
	id++;
      }
    }
  }
    

}

#undef __FUNCT__
#define __FUNCT__ "density_filter"
void density_filter(MPI_Comm comm, Mat *Qout, PetscInt nr, PetscInt ncells, PetscInt nlayers, PetscReal fr, PetscReal sigma, PetscInt normalized)
{


  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);

  PetscPrintf(comm,"Creating the density filter. NOTE: the radius fr must be greater than 0. fr <= 1 means no filter.\n");

  PetscInt Nr=nr*ncells;
  PetscInt ncols=Nr*nlayers;
  PetscInt nrows=ncols;

  Mat Q;

  PetscInt box_size=2*ceil(fr)-1;

  MatCreate(comm,&Q);
  MatSetType(Q,MATRIX_TYPE);
  MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Q, box_size, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Q, box_size, PETSC_NULL, box_size, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Q, &ns, &ne);

  PetscInt *cols = (PetscInt *)malloc(box_size*sizeof(PetscInt));
  PetscScalar *weights = (PetscScalar *)malloc(box_size*sizeof(PetscScalar));

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ir=(k=i)%Nr;
    PetscInt il=(k/=Nr)%nlayers;

    PetscInt jr_min=ir-ceil(fr)+1;
    PetscInt jr_max=ir+ceil(fr);

    PetscInt ind=0;
    PetscScalar norm=0.0+PETSC_i*0.0;

    for(PetscInt jr=jr_min;jr<jr_max;jr++){

      PetscInt jjr,jjl;

      if(jr < 0)
	jjr = -jr; //jjr = jr + Nr;
      else if(jr >= Nr)
	jjr = 2*Nr - jr - 2;  //jjr = jr - Nr;
      else
	jjr = jr;

      jjl=il;

      PetscInt j=jjr+Nr*jjl;
      PetscReal dist2=pow(ir-jr,2);
      PetscReal sigma2=pow(sigma,2);
      cols[ind]=j;
      weights[ind]=exp(-dist2/sigma2)+PETSC_i*0.0;
      norm += weights[ind];
      ind++;

    }

    if(normalized==1)
      for(PetscInt j=0;j<ind;j++) weights[j]/=norm;

    MatSetValues(Q, 1, &i, ind, cols, weights, ADD_VALUES);

  }

  MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);

  free(cols);
  free(weights);

  *Qout = Q;

}
