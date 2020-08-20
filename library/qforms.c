#include "qforms.h"

void create_Qp(MPI_Comm comm, Mat *Qp_out,
	       PetscInt nr, PetscInt nz,
	       PetscInt npmlr0, PetscInt npmlr1, PetscInt npmlz0, PetscInt npmlz1,
	       PetscReal dr, PetscReal dz,
	       PetscReal omega,
	       PetscInt m,
	       PetscInt ir_lstart, PetscInt nr_mtr, PetscInt ir_gstart, PetscInt iz_mtr)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);

  PetscInt nc = 3;
  PetscInt nrows = nc*nr*nz;
  PetscInt ncols = nc*nr*nz;

  Mat Rp;

  MatCreate(comm,&Rp);
  MatSetType(Rp,MATRIX_TYPE);
  MatSetSizes(Rp,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Rp, 1, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Rp, 1, PETSC_NULL, 1, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Rp, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ir=(k=i)%nr;
    PetscInt iz=(k/=nr)%nz;
    PetscInt ic=(k/=nz)%nc;

    if(ir>=ir_lstart && ir<ir_lstart+nr_mtr && iz==iz_mtr && ic<2){

      PetscReal r=(ir_gstart + ir - ir_lstart +0.5)*dr;
      PetscInt jr=ir;
      PetscInt jz=iz;
      PetscInt jc;
      PetscScalar val;
      if(ic==0)
	jc=1, val= 2*M_PI*r*dr/(PETSC_i*omega);
      else
	jc=0, val=-2*M_PI*r*dr/(PETSC_i*omega);

      PetscInt j=jr+nr*jz+nr*nz*jc;

      MatSetValues(Rp, 1, &i, 1, &j, &val, ADD_VALUES);
      
    }
    
  }

  MatAssemblyBegin(Rp, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Rp, MAT_FINAL_ASSEMBLY);
  
  Mat Sh;
  syncH(comm, &Sh, nr,nz);

  Mat De;
  PetscReal R0=(ir_gstart + 0.5)*dr;
  create_De(comm, &De,
	    nr,nz,
	    npmlr0,npmlr1,npmlz0,npmlz1,
	    dr,dz,R0,
	    omega,
	    m);

  Mat Q;
  MatMatMatMult(Rp,Sh,De, MAT_INITIAL_MATRIX, 16.0/(1.0+4.0+4.0), &Q);

  Mat Qdag;
  MatHermitianTranspose(Q,MAT_INITIAL_MATRIX,&Qdag);

  Mat Se;
  syncE(comm, &Se, nr,nz);

  Mat Qp;
  MatMatMult(Qdag,Se, MAT_INITIAL_MATRIX, 64.0/(16.0+4.0), &Qp);

  *Qp_out = Qp;

  MatDestroy(&Rp);
  MatDestroy(&Sh);
  MatDestroy(&De);
  MatDestroy(&Q);
  MatDestroy(&Qdag);
  MatDestroy(&Se);
  
}

void gforms(MPI_Comm comm, Vec gex, Vec gey, Vec ghx, Vec ghy,
	    PetscInt nr, PetscInt nz,
	    PetscInt npmlr0, PetscInt npmlr1, PetscInt npmlz0, PetscInt npmlz1,
	    PetscReal dr, PetscReal dz,
	    PetscReal omega,
	    PetscInt m,
	    PetscInt ir_lstart, PetscInt nr_mtr, PetscInt ir_gstart, PetscInt iz_mtr,
	    PetscReal xfar, PetscReal yfar, PetscReal zfar, PetscReal znear,
	    PetscReal mu, PetscReal eps)
{

  const PetscInt MAXEVAL_green=1000000;
  const PetscReal ABSERR_green=1e-6;
  const PetscReal RELERR_green=1e-6;
  
  Vec fexe,fexh,feye,feyh,fhxe,fhxh,fhye,fhyh;
  VecDuplicate(gex,&fexe);
  VecDuplicate(gex,&fexh);
  VecDuplicate(gex,&feye);
  VecDuplicate(gex,&feyh);
  VecDuplicate(gex,&fhxe);
  VecDuplicate(gex,&fhxh);
  VecDuplicate(gex,&fhye);
  VecDuplicate(gex,&fhyh);
  
  PetscScalar *exer = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *exet = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *exhr = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *exht = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *eyer = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *eyet = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *eyhr = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *eyht = (PetscScalar *)malloc(nr*sizeof(PetscScalar));

  PetscScalar *hxer = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hxet = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hxhr = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hxht = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hyer = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hyet = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hyhr = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  PetscScalar *hyht = (PetscScalar *)malloc(nr*sizeof(PetscScalar));
  
  PetscReal theta0=0;
  PetscReal theta1=2*M_PI;
  for(PetscInt ir=0;ir<nr;ir++){

    if(ir>=ir_lstart && ir<ir_lstart+nr_mtr){
      
      PetscReal irnear = ir_gstart + ir - ir_lstart;
      PetscReal params[10]={xfar,yfar,zfar,znear, irnear, dr, m, omega,mu,eps};

      PetscReal err[48];
      PetscReal EH[48];

      hcubature(48, fE, params,
		1, &theta0, &theta1,
		MAXEVAL_green, ABSERR_green, RELERR_green,
		ERROR_L2,
		EH, err);

      exer[ir]=EH[0] + PETSC_i * EH[0+24];
      exet[ir]=EH[1] + PETSC_i * EH[1+24];
      exhr[ir]=EH[2] + PETSC_i * EH[2+24];
      exht[ir]=EH[3] + PETSC_i * EH[3+24];
      eyer[ir]=EH[4] + PETSC_i * EH[4+24];
      eyet[ir]=EH[5] + PETSC_i * EH[5+24];
      eyhr[ir]=EH[6] + PETSC_i * EH[6+24];
      eyht[ir]=EH[7] + PETSC_i * EH[7+24];
      
      hxer[ir]=EH[12] + PETSC_i * EH[12+24];
      hxet[ir]=EH[13] + PETSC_i * EH[13+24];
      hxhr[ir]=EH[14] + PETSC_i * EH[14+24];
      hxht[ir]=EH[15] + PETSC_i * EH[15+24];
      hyer[ir]=EH[16] + PETSC_i * EH[16+24];
      hyet[ir]=EH[17] + PETSC_i * EH[17+24];
      hyhr[ir]=EH[18] + PETSC_i * EH[18+24];
      hyht[ir]=EH[19] + PETSC_i * EH[19+24];
      
    }else{

      exer[ir]=0;
      exet[ir]=0;
      exhr[ir]=0;
      exht[ir]=0;
      eyer[ir]=0;
      eyet[ir]=0;
      eyhr[ir]=0;
      eyht[ir]=0;
      
      hxer[ir]=0;
      hxet[ir]=0;
      hxhr[ir]=0;
      hxht[ir]=0;
      hyer[ir]=0;
      hyet[ir]=0;
      hyhr[ir]=0;
      hyht[ir]=0;

    }

  }

  vecfill_zslice(fexe, exer,exet, nr,nz, iz_mtr);
  vecfill_zslice(fexh, exhr,exht, nr,nz, iz_mtr);
  vecfill_zslice(feye, eyer,eyet, nr,nz, iz_mtr);
  vecfill_zslice(feyh, eyhr,eyht, nr,nz, iz_mtr);
  vecfill_zslice(fhxe, hxer,hxet, nr,nz, iz_mtr);
  vecfill_zslice(fhxh, hxhr,hxht, nr,nz, iz_mtr);
  vecfill_zslice(fhye, hyer,hyet, nr,nz, iz_mtr);
  vecfill_zslice(fhyh, hyhr,hyht, nr,nz, iz_mtr);

  Mat Se,Sh;
  syncE(comm, &Se, nr,nz);
  syncH(comm, &Sh, nr,nz);

  Mat De;
  PetscReal R0=(ir_gstart + 0.5)*dr;
  create_De(comm, &De,
	    nr,nz,
	    npmlr0,npmlr1,npmlz0,npmlz1,
	    dr,dz,R0,
	    omega,
	    m);
  MatScale(De,1.0/(PETSC_i * omega));
  Mat Qh;
  MatMatMult(Sh,De, MAT_INITIAL_MATRIX, 16.0/(4.0+4.0), &Qh);

  Vec tmp;
  VecDuplicate(gex,&tmp);
  
  MatMultTranspose(Se,fexe,gex);
  MatMultTranspose(Qh,fexh,tmp);
  VecAXPY(gex,1.0,tmp);

  MatMultTranspose(Se,feye,gey);
  MatMultTranspose(Qh,feyh,tmp);
  VecAXPY(gey,1.0,tmp);

  MatMultTranspose(Se,fhxe,ghx);
  MatMultTranspose(Qh,fhxh,tmp);
  VecAXPY(ghx,1.0,tmp);

  MatMultTranspose(Se,fhye,ghy);
  MatMultTranspose(Qh,fhyh,tmp);
  VecAXPY(ghy,1.0,tmp);
  
  MatDestroy(&Se);
  MatDestroy(&Sh);
  MatDestroy(&De);
  MatDestroy(&Qh);
  VecDestroy(&tmp);
  VecDestroy(&fexe);
  VecDestroy(&fexh);
  VecDestroy(&feye);
  VecDestroy(&feyh);
  VecDestroy(&fhxe);
  VecDestroy(&fhxh);
  VecDestroy(&fhye);
  VecDestroy(&fhyh);
  
  free(exer);
  free(exet);
  free(exhr);
  free(exht);
  free(eyer);
  free(eyet);
  free(eyhr);
  free(eyht);

  free(hxer);
  free(hxet);
  free(hxhr);
  free(hxht);
  free(hyer);
  free(hyet);
  free(hyhr);
  free(hyht);

}
