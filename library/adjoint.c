#include "adjoint.h"

void strehlpieces(MPI_Comm subcomm, PetscInt Nr, PetscInt Nz, PetscScalar *_edof, Mat A,
		  Mat DDe, Vec epsDiff, Vec epsBkg, PetscReal omega,
		  KSP ksp, int *its, int maxit,
		  PetscScalar *Jmr, PetscScalar *Jmt, PetscInt iz_src,
		  Vec gex, Vec gey, Vec ghx, Vec ghy, Mat Qp,
		  PetscScalar *Ex, PetscScalar *Ey, PetscScalar *Hx, PetscScalar *Hy, PetscScalar *Pz, 
		  PetscScalar *Ex_grad, PetscScalar *Ey_grad,
		  PetscScalar *Hx_grad, PetscScalar *Hy_grad,
		  PetscScalar *Pz_grad,
		  PetscInt m, Vec sigmahat,
		  Vec gex_minus, Vec gey_minus, Vec ghx_minus, Vec ghy_minus, Mat Qp_minus)
{

  Vec edof,eps;
  MatCreateVecs(A,&edof,&eps);
  array2mpi(_edof,SCAL,edof);
  MatMult(A,edof,eps);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);

  Mat M;
  MatDuplicate(DDe,MAT_COPY_VALUES,&M);
  VecScale(eps,-omega*omega);
  MatDiagonalSet(M,eps,ADD_VALUES);

  Vec b;
  VecDuplicate(eps,&b);
  VecSet(b,0);
  vecfill_zslice(b,Jmr,Jmt,Nr,Nz,iz_src);
  VecScale(b,PETSC_i*omega);

  Vec x;
  VecDuplicate(eps,&x);
  SolveMatrixDirect(subcomm,ksp,M,b,x,its,maxit);

  Vec u,tmp,grad;
  VecDuplicate(eps,&u);
  VecDuplicate(eps,&tmp);
  VecDuplicate(eps,&grad);
  PetscInt nedof;
  VecGetSize(edof,&nedof);

  if(Ex && gex && Ex_grad){
    //Compute Ex = gex . x and its grad
    VecTDot(gex,x,Ex);

    KSPSolveTranspose(ksp,gex,u);
    VecPointwiseMult(grad,u,x);
    VecScale(grad,omega*omega);
    VecPointwiseMult(grad,grad,epsDiff);
    MatMultTranspose(A,grad,edof);
    mpi2array(edof,Ex_grad,SCAL,nedof);

    if(abs(m)>0 && gex_minus){
      Vec gminus;
      VecDuplicate(x,&gminus);
    
      PetscScalar Ex_minus;
      PetscScalar *grad_minus=(PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    
      //Ex_minus
      VecPointwiseMult(gminus,gex_minus,sigmahat);
      VecTDot(gminus,x,&Ex_minus);
      *Ex=*Ex+Ex_minus;


      KSPSolveTranspose(ksp,gminus,u);
      VecPointwiseMult(grad,u,x);
      VecScale(grad,omega*omega);
      VecPointwiseMult(grad,grad,epsDiff);
      MatMultTranspose(A,grad,edof);
      mpi2array(edof,grad_minus,SCAL,nedof);
      for(PetscInt i=0;i<nedof;i++)
	Ex_grad[i] += grad_minus[i];

      VecDestroy(&gminus);
      free(grad_minus);
    }
    
  }

  if(Ey && gey && Ey_grad){
    //Compute Ey = gey . x and its grad
    VecTDot(gey,x,Ey);

    KSPSolveTranspose(ksp,gey,u);
    VecPointwiseMult(grad,u,x);
    VecScale(grad,omega*omega);
    VecPointwiseMult(grad,grad,epsDiff);
    MatMultTranspose(A,grad,edof);
    mpi2array(edof,Ey_grad,SCAL,nedof);  

    if(abs(m)>0 && gey_minus){
      Vec gminus;
      VecDuplicate(x,&gminus);
    
      PetscScalar Ey_minus;
      PetscScalar *grad_minus=(PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    
      //Ey_minus
      VecPointwiseMult(gminus,gey_minus,sigmahat);
      VecTDot(gminus,x,&Ey_minus);
      *Ey=*Ey+Ey_minus;

      KSPSolveTranspose(ksp,gminus,u);
      VecPointwiseMult(grad,u,x);
      VecScale(grad,omega*omega);
      VecPointwiseMult(grad,grad,epsDiff);
      MatMultTranspose(A,grad,edof);
      mpi2array(edof,grad_minus,SCAL,nedof);
      for(PetscInt i=0;i<nedof;i++)
	Ey_grad[i] += grad_minus[i];

      VecDestroy(&gminus);
      free(grad_minus);
    }
    
  }

  if(Hx && ghx && Hx_grad){
    //Compute Hx = ghx . x and its grad
    VecTDot(ghx,x,Hx);

    KSPSolveTranspose(ksp,ghx,u);
    VecPointwiseMult(grad,u,x);
    VecScale(grad,omega*omega);
    VecPointwiseMult(grad,grad,epsDiff);
    MatMultTranspose(A,grad,edof);
    mpi2array(edof,Hx_grad,SCAL,nedof);

    if(abs(m)>0 && ghx_minus){
      Vec gminus;
      VecDuplicate(x,&gminus);
    
      PetscScalar Hx_minus;
      PetscScalar *grad_minus=(PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    
      //Hx_minus
      VecPointwiseMult(gminus,ghx_minus,sigmahat);
      VecTDot(gminus,x,&Hx_minus);
      *Hx=*Hx+Hx_minus;

      KSPSolveTranspose(ksp,gminus,u);
      VecPointwiseMult(grad,u,x);
      VecScale(grad,omega*omega);
      VecPointwiseMult(grad,grad,epsDiff);
      MatMultTranspose(A,grad,edof);
      mpi2array(edof,grad_minus,SCAL,nedof);
      for(PetscInt i=0;i<nedof;i++)
	Hx_grad[i] += grad_minus[i];

      VecDestroy(&gminus);
      free(grad_minus);
    }

  }

  if(Hy && ghy && Hy_grad){
    //Compute Hy = ghy . x and its grad
    VecTDot(ghy,x,Hy);

    KSPSolveTranspose(ksp,ghy,u);
    VecPointwiseMult(grad,u,x);
    VecScale(grad,omega*omega);
    VecPointwiseMult(grad,grad,epsDiff);
    MatMultTranspose(A,grad,edof);
    mpi2array(edof,Hy_grad,SCAL,nedof);  

    if(abs(m)>0 && ghy_minus){
      Vec gminus;
      VecDuplicate(x,&gminus);
    
      PetscScalar Hy_minus;
      PetscScalar *grad_minus=(PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    
      //Hy_minus
      VecPointwiseMult(gminus,ghy_minus,sigmahat);
      VecTDot(gminus,x,&Hy_minus);
      *Hy=*Hy+Hy_minus;

      KSPSolveTranspose(ksp,gminus,u);
      VecPointwiseMult(grad,u,x);
      VecScale(grad,omega*omega);
      VecPointwiseMult(grad,grad,epsDiff);
      MatMultTranspose(A,grad,edof);
      mpi2array(edof,grad_minus,SCAL,nedof);
      for(PetscInt i=0;i<nedof;i++)
	Hy_grad[i] += grad_minus[i];

      VecDestroy(&gminus);
      free(grad_minus);
    }

  }

  if(Pz && Qp && Pz_grad){
    //Compute Pz = x* . Qp . x and its grad 
    Vec xconj;
    VecDuplicate(eps,&xconj);
    VecCopy(x,xconj);
    VecConjugate(xconj);
    MatMult(Qp,x,tmp);
    VecTDot(xconj,tmp,Pz);
    
    /*
    Mat conjQp;
    MatDuplicate(Qp,MAT_COPY_VALUES,&conjQp);
    MatConjugate(conjQp);
    MatMult(conjQp,xconj,tmp);
    MatDestroy(&conjQp);
    */

    VecConjugate(tmp);
    MatMultTranspose(Qp,xconj,grad);
    VecAXPY(tmp,1.0,grad);

    KSPSolveTranspose(ksp,tmp,u);
    VecPointwiseMult(grad,u,x);
    VecScale(grad,omega*omega);
    VecPointwiseMult(grad,grad,epsDiff);
    MatMultTranspose(A,grad,edof);
    mpi2array(edof,Pz_grad,SCAL,nedof);  

    if(abs(m)>0 && Qp_minus){
      Vec gminus;
      VecDuplicate(x,&gminus);
    
      PetscScalar Pz_minus;
      PetscScalar *grad_minus=(PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    
      //Pz_minus
      Mat Q2;
      MatDuplicate(Qp_minus,MAT_COPY_VALUES,&Q2);    
      MatDiagonalScale(Q2,sigmahat,sigmahat);
      MatMult(Q2,x,tmp);
      VecTDot(xconj,tmp,&Pz_minus);
      *Pz=*Pz+Pz_minus;

      /*
      Mat conjQ2;
      MatDuplicate(Q2,MAT_COPY_VALUES,&conjQ2);
      MatConjugate(conjQ2);
      MatMult(conjQ2,xconj,tmp);
      MatDestroy(&conjQ2);
      */

      VecConjugate(tmp);
      MatMultTranspose(Q2,xconj,grad);
      VecAXPY(tmp,1.0,grad);
    
      KSPSolveTranspose(ksp,tmp,u);
      VecPointwiseMult(grad,u,x);
      VecScale(grad,omega*omega);
      VecPointwiseMult(grad,grad,epsDiff);
      MatMultTranspose(A,grad,edof);
      mpi2array(edof,grad_minus,SCAL,nedof);      
      for(PetscInt i=0;i<nedof;i++)
	Pz_grad[i] += grad_minus[i];

      VecDestroy(&gminus);
      free(grad_minus);
      MatDestroy(&Q2);
    }

    VecDestroy(&xconj);

  }

    
  VecDestroy(&edof);
  VecDestroy(&eps);
  VecDestroy(&b);
  VecDestroy(&x);
  VecDestroy(&u);
  VecDestroy(&tmp);
  VecDestroy(&grad);
  MatDestroy(&M);


}
