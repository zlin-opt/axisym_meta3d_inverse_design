#include "maxwell.h"

#define alpha 4
#define Refl cexp(-16)
#define n0bkg 1.0

#define R0tol 1e-8

#undef __FUNCT__
#define __FUNCT__ "pml_s"
PetscScalar pml_s(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega)
{

  PetscReal p=i*delta;
  PetscReal ps=Npml0*delta;
  PetscReal pb=(N-Npml1)*delta;

  PetscReal d0=Npml0*delta;
  PetscReal d1=Npml1*delta;
  PetscReal l,Lpml;
  if (p<ps)
    l=ps-p,Lpml=d0; 
  else if (p>pb)
    l=p-pb,Lpml=d1;
  else
    l=0,Lpml=1;

  PetscReal lnR=log(Refl);
  PetscReal sigma=-(alpha+1)*lnR/(2*n0bkg*omega*Lpml);

  PetscScalar s = 1.0 + PETSC_i * sigma * pow(l/Lpml,alpha);

  return s;

}

#undef __FUNCT__
#define __FUNCT__ "pml_p"
PetscScalar pml_p(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega)
{

  PetscReal p=i*delta;
  PetscReal ps=Npml0*delta;
  PetscReal pb=(N-Npml1)*delta;
  
  PetscReal d0=Npml0*delta;
  PetscReal d1=Npml1*delta;
  PetscReal l,Lpml;
  PetscInt sign;
  if (p<ps)
    l=ps-p,Lpml=d0,sign=-1; 
  else if (p>pb)
    l=p-pb,Lpml=d1,sign=1;
  else
    l=0,Lpml=1,sign=0;

  PetscReal lnR=log(Refl);
  PetscReal sigma=-(alpha+1)*lnR/(2*n0bkg*omega*Lpml);

  PetscScalar pprime = p + PETSC_i * sign * (sigma*Lpml/(alpha+1)) * pow(l/Lpml,alpha+1);

  return pprime;

}

//Note that the indexing chosen as ir,iz,ic in the order of the fastest to slowest

#undef __FUNCT__
#define __FUNCT__ "create_De"
void create_De(MPI_Comm comm, Mat *De_out,
	       PetscInt Nr, PetscInt Nz,
	       PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
	       PetscReal dr, PetscReal dz, PetscReal R0,
	       PetscReal omega,
	       PetscInt m)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt Nc = 3;
  PetscInt nrows = Nc*Nr*Nz;
  PetscInt ncols = Nc*Nr*Nz;
  
  //PetscPrintf(comm,"Creating the matrix De (which operates on E fields).\n");

  Mat De;

  MatCreate(comm,&De);
  MatSetType(De,MATRIX_TYPE);
  MatSetSizes(De,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(De, 4, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(De, 4, PETSC_NULL, 4, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(De, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    
    PetscInt k;
    PetscInt ir=(k=i)%Nr;
    PetscInt iz=(k/=Nr)%Nz;
    PetscInt ic=(k/=Nz)%Nc;

    //Hr -> Et,Ez
    if(ic==0){

      PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.5, dz, omega);
      PetscScalar rp=pml_p(Nr, Npmlr0,Npmlr1, ir+0.0, dr, omega);
      PetscInt jc0=1,    jc1=1,  jc2=2;
      PetscInt jr0=ir,   jr1=ir, jr2=ir;
      PetscInt jz0=iz+1, jz1=iz, jz2=iz;

      PetscScalar val0 = -1.0/(sz*dz);
      PetscScalar val1 = +1.0/(sz*dz);
      PetscScalar val2 = (R0+creal(rp)>R0tol) ? PETSC_i * m/(R0+rp) : 0 ;      
      if (iz==Nz-1)                       jz0=Nz-1, val0=0;
      if (ir==0 && R0<R0tol && abs(m)==1) jr2=1,                    val2=PETSC_i*m/dr;
      if (ir==0 && R0<R0tol && abs(m)!=1)           val0=0, val1=0, val2=0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt mcols=3;
      PetscInt jcols[3]={j0,j1,j2};
      PetscScalar vals[3]={val0,val1,val2};
      
      MatSetValues(De, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    //Ht -> Er,Ez
    if(ic==1){

      PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.5, dz, omega);
      PetscScalar sr=pml_s(Nr, Npmlr0,Npmlr1, ir+0.5, dr, omega);
      PetscInt jc0=0,    jc1=0,  jc2=2,    jc3=2;
      PetscInt jr0=ir,   jr1=ir, jr2=ir+1, jr3=ir;
      PetscInt jz0=iz+1, jz1=iz, jz2=iz,   jz3=iz;

      PetscScalar val0 = +1.0/(sz*dz);
      PetscScalar val1 = -1.0/(sz*dz);
      PetscScalar val2 = -1.0/(sr*dr);
      PetscScalar val3 = +1.0/(sr*dr);
      if (ir==Nr-1) jr2=Nr-1, val2=0;
      if (iz==Nz-1) jz0=Nz-1, val0=0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt j3=jr3 + Nr*jz3 + Nr*Nz*jc3;
      PetscInt mcols=4;
      PetscInt jcols[4]={j0,j1,j2,j3};
      PetscScalar vals[4]={val0,val1,val2,val3};
      
      MatSetValues(De, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    //Hz -> Er,Et
    if(ic==2){

      PetscScalar sr =pml_s(Nr, Npmlr0,Npmlr1, ir+0.5, dr, omega);
      PetscScalar rp1=pml_p(Nr, Npmlr0,Npmlr1, ir+0.0, dr, omega);
      PetscScalar rp2=pml_p(Nr, Npmlr0,Npmlr1, ir+0.5, dr, omega);
      PetscScalar rp3=pml_p(Nr, Npmlr0,Npmlr1, ir+1.0, dr, omega);
      PetscInt jc0=0,  jc1=1,    jc2=1;
      PetscInt jr0=ir, jr1=ir+1, jr2=ir;
      PetscInt jz0=iz, jz1=iz,   jz2=iz;

      PetscScalar val0 = -PETSC_i * m/(R0+rp2);
      PetscScalar val1 = +(R0+rp3)/(sr*(R0+rp2)*dr);
      PetscScalar val2 = -(R0+rp1)/(sr*(R0+rp2)*dr);
      if (ir==Nr-1) jr1=Nr-1, val1=0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt mcols=3;
      PetscInt jcols[3]={j0,j1,j2};
      PetscScalar vals[3]={val0,val1,val2};
      
      MatSetValues(De, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }
	
  }

  MatAssemblyBegin(De, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(De, MAT_FINAL_ASSEMBLY);

  *De_out = De;

}


#undef __FUNCT__
#define __FUNCT__ "create_Dh"
void create_Dh(MPI_Comm comm, Mat *Dh_out,
	       PetscInt Nr, PetscInt Nz,
	       PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
	       PetscReal dr, PetscReal dz, PetscReal R0,
	       PetscReal omega,
	       PetscInt m)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt Nc = 3;
  PetscInt nrows = Nc*Nr*Nz;
  PetscInt ncols = Nc*Nr*Nz;

  //PetscPrintf(comm,"Creating the matrix Dh (which operates on H fields).\n");

  Mat Dh;

  MatCreate(comm,&Dh);
  MatSetType(Dh,MATRIX_TYPE);
  MatSetSizes(Dh,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Dh, 4, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Dh, 4, PETSC_NULL, 4, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Dh, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){


    PetscInt k;
    PetscInt ir=(k=i)%Nr;
    PetscInt iz=(k/=Nr)%Nz;
    PetscInt ic=(k/=Nz)%Nc;

    //Er -> Ht,Hz
    if(ic==0){

      PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.0, dz, omega);
      PetscScalar rp=pml_p(Nr, Npmlr0,Npmlr1, ir+0.5, dr, omega);
      PetscInt jc0=1,  jc1=1,    jc2=2;
      PetscInt jr0=ir, jr1=ir,   jr2=ir;
      PetscInt jz0=iz, jz1=iz-1, jz2=iz;

      PetscScalar val0 = -1.0/(sz*dz);
      PetscScalar val1 = +1.0/(sz*dz);
      PetscScalar val2 = +PETSC_i * m/(R0+rp);
      if (iz==0) jz1=0, val1=0;
 
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt mcols=3;
      PetscInt jcols[3]={j0,j1,j2};
      PetscScalar vals[3]={val0,val1,val2};

      MatSetValues(Dh, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    //Et -> Hr,Hz
    if(ic==1){

      PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.0, dz, omega);
      PetscScalar sr=pml_s(Nr, Npmlr0,Npmlr1, ir+0.0, dr, omega);
      PetscInt jc0=0,  jc1=0,    jc2=2,  jc3=2;
      PetscInt jr0=ir, jr1=ir,   jr2=ir, jr3=ir-1;
      PetscInt jz0=iz, jz1=iz-1, jz2=iz, jz3=iz;

      PetscScalar val0 = +1.0/(sz*dz);
      PetscScalar val1 = -1.0/(sz*dz);
      PetscScalar val2 = -1.0/(sr*dr);
      PetscScalar val3 = +1.0/(sr*dr);
      if (iz==0                          ) jz1=0,         val1=0;
      if (ir==0 && R0>=R0tol             ) jr3=0,                               val3=0;
      if (ir==0 && R0<R0tol  && abs(m)==1) jr3=0,                 val2=-2.0/dr, val3=0;
      if (ir==0 && R0<R0tol  && abs(m)!=1) jr3=0, val0=0, val1=0, val2=0,       val3=0;
      

      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt j3=jr3 + Nr*jz3 + Nr*Nz*jc3;
      PetscInt mcols=4;
      PetscInt jcols[4]={j0,j1,j2,j3};
      PetscScalar vals[4]={val0,val1,val2,val3};

      MatSetValues(Dh, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    //Ez -> Hr,Ht
    if(ic==2){

      PetscScalar sr =pml_s(Nr, Npmlr0,Npmlr1, ir+0.0, dr, omega);
      PetscScalar rp1=pml_p(Nr, Npmlr0,Npmlr1, ir-0.5, dr, omega);
      PetscScalar rp2=pml_p(Nr, Npmlr0,Npmlr1, ir+0.0, dr, omega);
      PetscScalar rp3=pml_p(Nr, Npmlr0,Npmlr1, ir+0.5, dr, omega);
      PetscInt jc0=0,  jc1=1,  jc2=1;
      PetscInt jr0=ir, jr1=ir, jr2=ir-1;
      PetscInt jz0=iz, jz1=iz, jz2=iz;

      PetscScalar val0 =  (R0+creal(rp2)>R0tol) ? -PETSC_i * m/(R0+rp2) : 0 ;
      PetscScalar val1 = +(R0+rp3)/(sr*(R0+rp2)*dr);
      PetscScalar val2 = -(R0+rp1)/(sr*(R0+rp2)*dr);
      if (ir==0 && R0>=R0tol        ) jr2=0,                      val2=0;
      if (ir==0 && R0<R0tol  && m==0) jr2=0, val0=0, val1=4.0/dr, val2=0;
      if (ir==0 && R0<R0tol  && m!=0) jr2=0, val0=0, val1=0,      val2=0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt mcols=3;
      PetscInt jcols[3]={j0,j1,j2};
      PetscScalar vals[3]={val0,val1,val2};

      MatSetValues(Dh, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

  }

  MatAssemblyBegin(Dh, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Dh, MAT_FINAL_ASSEMBLY);

  *Dh_out = Dh;

}


#undef __FUNCT__
#define __FUNCT__ "create_DDe"
void create_DDe(MPI_Comm comm, Mat *DDe_out,
	        PetscInt Nr, PetscInt Nz,
	        PetscInt Npmlr0, PetscInt Npmlr1, PetscInt Npmlz0, PetscInt Npmlz1,
	        PetscReal dr, PetscReal dz, PetscReal R0,
	        PetscReal omega,
	        PetscInt m)
{

  //PetscPrintf(comm,"Creating the matrix Dh.De (the double curl operator that acts on E fields).\n");  
  
  Mat De,Dh;
  create_De(comm, &De, Nr,Nz, Npmlr0,Npmlr1,Npmlz0,Npmlz1, dr,dz, R0, omega, m);
  create_Dh(comm, &Dh, Nr,Nz, Npmlr0,Npmlr1,Npmlz0,Npmlz1, dr,dz, R0, omega, m);
  Mat DDe;
  MatMatMult(Dh, De, MAT_INITIAL_MATRIX, 13.0/(4.0+4.0), &DDe);

  *DDe_out = DDe;

  MatDestroy(&De);
  MatDestroy(&Dh);

}

/*******************************************
 *******************************************
 *******************************************/
//create matrices that sync the fields to [i+1/2,j+1/2]
//Note that the indexing chosen as ir,iz,ic in the order of the fastest to slowest

#undef __FUNCT__
#define __FUNCT__ "syncE"
void syncE(MPI_Comm comm, Mat *Ae_out, PetscInt Nr, PetscInt Nz)
{

  PetscInt Nc = 3;
  PetscInt nrows = Nc*Nr*Nz;
  PetscInt ncols = Nc*Nr*Nz;

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  //PetscPrintf(comm,"Creating the matrix Ae to sync the E fields.\n");

  Mat Ae;

  MatCreate(comm,&Ae);
  MatSetType(Ae,MATRIX_TYPE);
  MatSetSizes(Ae,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Ae, 4, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Ae, 4, PETSC_NULL, 4, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Ae, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    
    PetscInt k;
    PetscInt ir=(k=i)%Nr;
    PetscInt iz=(k/=Nr)%Nz;
    PetscInt ic=(k/=Nz)%Nc;

    if(ic==0){

      PetscInt jc0=0,  jc1=0;
      PetscInt jr0=ir, jr1=ir;
      PetscInt jz0=iz, jz1=iz+1;
      PetscScalar val0=0.5, val1=0.5;
      if (iz==Nz-1) jz1=Nz-1, val1=0.0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt mcols=2;
      PetscInt jcols[2]={j0,j1};
      PetscScalar vals[2]={val0,val1};
      
      MatSetValues(Ae, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    if(ic==1){

      PetscInt jc0=1,  jc1=1,    jc2=1,    jc3=1;
      PetscInt jr0=ir, jr1=ir+1, jr2=ir,   jr3=ir+1;
      PetscInt jz0=iz, jz1=iz,   jz2=iz+1, jz3=iz+1;
      PetscScalar val0=0.25, val1=0.25, val2=0.25, val3=0.25;
      if (ir==Nr-1) jr1=Nr-1, jr3=Nr-1, val1=0.0, val3=0.0;
      if (iz==Nz-1) jz2=Nz-1, jz3=Nz-1, val2=0.0, val3=0.0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt j2=jr2 + Nr*jz2 + Nr*Nz*jc2;
      PetscInt j3=jr3 + Nr*jz3 + Nr*Nz*jc3;
      PetscInt mcols=4;
      PetscInt jcols[4]={j0,j1,j2,j3};
      PetscScalar vals[4]={val0,val1,val2,val3};
      
      MatSetValues(Ae, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    if(ic==2){

      PetscInt jc0=2,  jc1=2;
      PetscInt jr0=ir, jr1=ir+1;
      PetscInt jz0=iz, jz1=iz;
      PetscScalar val0=0.5, val1=0.5;
      if (ir==Nr-1) jr1=Nr-1, val1=0.0;
      
      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt mcols=2;
      PetscInt jcols[2]={j0,j1};
      PetscScalar vals[2]={val0,val1};
      
      MatSetValues(Ae, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }
	
  }

  MatAssemblyBegin(Ae, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ae, MAT_FINAL_ASSEMBLY);

  *Ae_out = Ae;

}


#undef __FUNCT__
#define __FUNCT__ "syncH"
void syncH(MPI_Comm comm, Mat *Ah_out, PetscInt Nr, PetscInt Nz)
{

  PetscInt Nc = 3;
  PetscInt nrows = Nc*Nr*Nz;
  PetscInt ncols = Nc*Nr*Nz;

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  //PetscPrintf(comm,"Creating the matrix Ah to sync the H fields.\n");

  Mat Ah;

  MatCreate(comm,&Ah);
  MatSetType(Ah,MATRIX_TYPE);
  MatSetSizes(Ah,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Ah, 4, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Ah, 4, PETSC_NULL, 4, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Ah, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){


    PetscInt k;
    PetscInt ir=(k=i)%Nr;
    PetscInt iz=(k/=Nr)%Nz;
    PetscInt ic=(k/=Nz)%Nc;

    if(ic==0){

      PetscInt jc0=0,  jc1=0;
      PetscInt jr0=ir, jr1=ir+1;
      PetscInt jz0=iz, jz1=iz;
      PetscScalar val0=0.5, val1=0.5;
      if (ir==Nr-1) jr1=Nr-1, val1=0.0;

      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt mcols=2;
      PetscInt jcols[2]={j0,j1};
      PetscScalar vals[4]={val0,val1};

      MatSetValues(Ah, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    if(ic==1){

      PetscInt mcols=1;
      PetscInt jcols[1]={i};
      PetscScalar vals[1]={1.0};

      MatSetValues(Ah, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

    if(ic==2){

      PetscInt jc0=2,  jc1=2;
      PetscInt jr0=ir, jr1=ir;
      PetscInt jz0=iz, jz1=iz+1;
      PetscScalar val0=0.5, val1=0.5;
      if (iz==Nz-1) jz1=Nz-1, val1=0.0;

      PetscInt j0=jr0 + Nr*jz0 + Nr*Nz*jc0;
      PetscInt j1=jr1 + Nr*jz1 + Nr*Nz*jc1;
      PetscInt mcols=2;
      PetscInt jcols[2]={j0,j1};
      PetscScalar vals[2]={val0,val1};

      MatSetValues(Ah, 1, &i, mcols, jcols, vals, ADD_VALUES);

    }

  }

  MatAssemblyBegin(Ah, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ah, MAT_FINAL_ASSEMBLY);

  *Ah_out = Ah;

}
