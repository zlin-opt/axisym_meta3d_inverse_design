#include "pwsrc.h"

void Jm(PetscScalar *Jmr, PetscScalar *Jmt,
	PetscInt ir_start, PetscInt nr_segment, PetscReal dr, PetscReal dz,
	PetscInt m,
	PetscReal k, PetscReal alpha, PetscReal phi,
	PetscScalar Ax, PetscScalar Ay)
{

  
  for (PetscInt ir=0;ir<nr_segment;ir++){

    PetscScalar p1,p2,Pm,Qm;
    
    PetscReal r0 = (ir_start+ir+0.5)*dr; //Jr is positioned at ir+1/2 grid points
    p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*r0*sin(alpha));
    p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*r0*sin(alpha));
    Pm =         M_PI*(p1+p2);
    Qm = PETSC_i*M_PI*(p1-p2);
    Jmr[ir] = (1.0/(2.0*M_PI*dz)) * (Ax*Pm + Ay*Qm);

    PetscReal r1 = (ir_start+ir)*dr; //Jtheta is positioned at ir grid points
    p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*r1*sin(alpha));
    p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*r1*sin(alpha));
    Pm =         M_PI*(p1+p2);
    Qm = PETSC_i*M_PI*(p1-p2);
    Jmt[ir] = (1.0/(2.0*M_PI*dz)) * (Ay*Pm - Ax*Qm);

  }

}


void m_select(PetscInt num_m_max, PetscReal rmin, PetscReal rmax, PetscReal k, PetscReal phi, PetscReal alpha, PetscReal cutoff,  PetscInt *num_m, PetscInt **mlist)
{

  PetscReal *pqm_rmin = (PetscReal *)malloc(num_m_max*sizeof(PetscReal));
  PetscReal *pqm_rmax = (PetscReal *)malloc(num_m_max*sizeof(PetscReal));
  PetscInt m1 = -num_m_max/2;

  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscScalar p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*rmin*sin(alpha));
    PetscScalar p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*rmin*sin(alpha));
    PetscReal Pm = cabs(        M_PI*(p1+p2));
    PetscReal Qm = cabs(PETSC_i*M_PI*(p1-p2));
    if(Pm>=Qm)
      pqm_rmin[im]=Pm;
    else
      pqm_rmin[im]=Qm;
  }
  PetscReal max_pqm_rmin=find_max(pqm_rmin,num_m_max);

  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscScalar p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*rmax*sin(alpha));
    PetscScalar p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*rmax*sin(alpha));
    PetscReal Pm = cabs(        M_PI*(p1+p2));
    PetscReal Qm = cabs(PETSC_i*M_PI*(p1-p2));
    if(Pm>=Qm)
      pqm_rmax[im]=Pm;
    else
      pqm_rmax[im]=Qm;
  }
  PetscReal max_pqm_rmax=find_max(pqm_rmax,num_m_max);

  PetscInt tmpnum_m=0;
  PetscInt tmpmlist[num_m_max];
  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscReal pqr1 = pqm_rmin[im]/max_pqm_rmin;
    PetscReal pqr2 = pqm_rmax[im]/max_pqm_rmax;
    if(pqr1 > cutoff || pqr2 > cutoff){
      tmpmlist[tmpnum_m]=m;
      tmpnum_m++;
    }
  }

  *num_m=tmpnum_m;
  PetscInt *tmp=(PetscInt *)malloc(*num_m*sizeof(PetscInt));
  for(PetscInt i=0;i<*num_m;i++)
    tmp[i]=tmpmlist[i];
  *mlist = tmp;
  
}

void make_sigmahat(Vec sigmahat, PetscInt Nr, PetscInt Nz, PetscInt pol)
{

  PetscReal *_sigma = (PetscReal *)malloc(Nr*Nz*3*sizeof(PetscReal));
  for(PetscInt ic=0;ic<3;ic++){
    for(PetscInt iz=0;iz<Nz;iz++){
      for(PetscInt ir=0;ir<Nr;ir++){

	PetscInt i=ir+Nr*iz+Nr*Nz*ic;
	if(ic==0){
	  if(pol==0)
	    _sigma[i]=1;
	  else
	    _sigma[i]=-1;
	}else if(ic==1){
	  if(pol==0)
	    _sigma[i]=-1;
	  else
	    _sigma[i]=1;
	}else{
	  if(pol==0)
	    _sigma[i]=1;
	  else
	    _sigma[i]=-1;
	}

      }
    }
  }
  array2mpi(_sigma,REAL, sigmahat);

  free(_sigma);

}

void m_select_nonneg(PetscInt num_m_max, PetscReal rmin, PetscReal rmax, PetscReal k, PetscReal phi, PetscReal alpha, PetscReal cutoff,  PetscInt *num_m, PetscInt **mlist)
{

  PetscReal *pqm_rmin = (PetscReal *)malloc(num_m_max*sizeof(PetscReal));
  PetscReal *pqm_rmax = (PetscReal *)malloc(num_m_max*sizeof(PetscReal));
  PetscInt m1 = 0;

  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscScalar p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*rmin*sin(alpha));
    PetscScalar p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*rmin*sin(alpha));
    PetscReal Pm = cabs(        M_PI*(p1+p2));
    PetscReal Qm = cabs(PETSC_i*M_PI*(p1-p2));
    if(Pm>=Qm)
      pqm_rmin[im]=Pm;
    else
      pqm_rmin[im]=Qm;
  }
  PetscReal max_pqm_rmin=find_max(pqm_rmin,num_m_max);

  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscScalar p1 = cpow(PETSC_i,m+1) * cexp(-PETSC_i*(m+1)*phi) * jn(m+1,k*rmax*sin(alpha));
    PetscScalar p2 = cpow(PETSC_i,m-1) * cexp(-PETSC_i*(m-1)*phi) * jn(m-1,k*rmax*sin(alpha));
    PetscReal Pm = cabs(        M_PI*(p1+p2));
    PetscReal Qm = cabs(PETSC_i*M_PI*(p1-p2));
    if(Pm>=Qm)
      pqm_rmax[im]=Pm;
    else
      pqm_rmax[im]=Qm;
  }
  PetscReal max_pqm_rmax=find_max(pqm_rmax,num_m_max);

  PetscInt tmpnum_m=0;
  PetscInt tmpmlist[num_m_max];
  for(PetscInt im=0;im<num_m_max;im++){
    PetscInt m=m1+im;
    PetscReal pqr1 = pqm_rmin[im]/max_pqm_rmin;
    PetscReal pqr2 = pqm_rmax[im]/max_pqm_rmax;
    if(pqr1 > cutoff || pqr2 > cutoff){
      tmpmlist[tmpnum_m]=m;
      tmpnum_m++;
    }
  }

  *num_m=tmpnum_m;
  PetscInt *tmp=(PetscInt *)malloc(*num_m*sizeof(PetscInt));
  for(PetscInt i=0;i<*num_m;i++)
    tmp[i]=tmpmlist[i];
  *mlist = tmp;

}
