#include "greendyadic.h"

const PetscInt maxeval_green=10000;
const PetscReal abserr_green=1e-6;
const PetscReal relerr_green=1e-6;

PetscInt fE(unsigned ndim, const PetscReal *t, void *data,
	    unsigned fdim, PetscReal *fval)
{

  PetscReal theta = t[0];
  PetscReal *params = (PetscReal *)data;

  PetscReal x       = params[0];
  PetscReal y       = params[1];
  PetscReal z       = params[2];
  PetscReal z0      = params[3];
  PetscInt  ir      = (PetscInt)round(params[4]);
  PetscReal dr      = params[5];
  PetscInt  m       = (PetscInt)round(params[6]);
  PetscReal omega   = params[7];
  PetscReal mu      = params[8];
  PetscReal eps     = params[9];

  PetscReal k  = omega * sqrt(mu*eps);
  
  PetscReal r = (ir+0.5)*dr;
  PetscReal Rx = x - r*cos(theta);
  PetscReal Ry = y - r*sin(theta);
  PetscReal Rz = z - z0;
  PetscReal Rr = sqrt(Rx*Rx + Ry*Ry + Rz*Rz);

  PetscScalar Ik = PETSC_i*k;
  PetscScalar Iw = PETSC_i*omega;
  PetscReal k2 = pow(k,2);
  PetscReal Rr2 = pow(Rr,2);
  PetscReal Rr3 = pow(Rr,3);
  
  PetscScalar g  = cexp(PETSC_i * k *Rr)/(4*M_PI*Rr);
  PetscScalar gx = Rx * (-1 + PETSC_i * k * Rr)*cexp(PETSC_i * k * Rr)/(4*M_PI*Rr3);
  PetscScalar gy = Ry * (-1 + PETSC_i * k * Rr)*cexp(PETSC_i * k * Rr)/(4*M_PI*Rr3);
  PetscScalar gz = Rz * (-1 + PETSC_i * k * Rr)*cexp(PETSC_i * k * Rr)/(4*M_PI*Rr3);

  PetscScalar gxx = ( 3*pow(Rx,2)/Rr3 - 3*Ik*pow(Rx,2)/Rr2 - (1+pow(k*Rx,2))/Rr + Ik ) * cexp(PETSC_i * k * Rr)/(4*M_PI*Rr2);
  PetscScalar gyy = ( 3*pow(Ry,2)/Rr3 - 3*Ik*pow(Ry,2)/Rr2 - (1+pow(k*Ry,2))/Rr + Ik ) * cexp(PETSC_i * k * Rr)/(4*M_PI*Rr2);

  PetscScalar gxy = ( 3/Rr2 - 3*Ik/Rr - k2 ) * cexp(PETSC_i * k * Rr)*Rx*Ry/(4*M_PI*Rr3);
  PetscScalar gzx = ( 3/Rr2 - 3*Ik/Rr - k2 ) * cexp(PETSC_i * k * Rr)*Rx*Rz/(4*M_PI*Rr3);
  PetscScalar gzy = ( 3/Rr2 - 3*Ik/Rr - k2 ) * cexp(PETSC_i * k * Rr)*Ry*Rz/(4*M_PI*Rr3);
  PetscScalar gyx = gxy;

  PetscReal Exr=cos(theta);
  PetscReal Ext=-sin(theta);
  PetscReal Eyr=sin(theta);
  PetscReal Eyt=cos(theta);
  PetscReal Hxr=cos(theta);
  PetscReal Hxt=-sin(theta);
  PetscReal Hyr=sin(theta);
  PetscReal Hyt=cos(theta);

  PetscScalar mrfac = r*cexp(PETSC_i * m * theta)*dr;
  
  PetscScalar ExEr = (-mu/eps) * gz * Exr * mrfac;
  PetscScalar ExEt = (-mu/eps) * gz * Ext * mrfac;
  PetscScalar ExHr = mu * ( -Iw * g * Hyr - (Iw/k2) * gxx * Hyr + (Iw/k2) * gxy * Hxr ) * mrfac;
  PetscScalar ExHt = mu * ( -Iw * g * Hyt - (Iw/k2) * gxx * Hyt + (Iw/k2) * gxy * Hxt ) * mrfac; 

  PetscScalar EyEr = (-mu/eps) * gz * Eyr * mrfac;
  PetscScalar EyEt = (-mu/eps) * gz * Eyt * mrfac;
  PetscScalar EyHr = mu * ( +Iw * g * Hxr - (Iw/k2) * gyx * Hyr + (Iw/k2) * gyy * Hxr ) * mrfac;
  PetscScalar EyHt = mu * ( +Iw * g * Hxt - (Iw/k2) * gyx * Hyt + (Iw/k2) * gyy * Hxt ) * mrfac; 

  PetscScalar EzEr = (mu/eps) * (gx * Exr + gy * Eyr) * mrfac;
  PetscScalar EzEt = (mu/eps) * (gx * Ext + gy * Eyt) * mrfac;
  PetscScalar EzHr = mu * ( -(Iw/k2) * gzx * Hyr + (Iw/k2) * gzy * Hxr ) * mrfac;
  PetscScalar EzHt = mu * ( -(Iw/k2) * gzx * Hyt + (Iw/k2) * gzy * Hxt ) * mrfac;

  PetscScalar HxHr = (-eps/mu) * gz * Hxr * mrfac;
  PetscScalar HxHt = (-eps/mu) * gz * Hxt * mrfac;
  PetscScalar HxEr = eps * ( +Iw * g * Eyr + (Iw/k2) * gxx * Eyr - (Iw/k2) * gxy * Exr ) * mrfac;
  PetscScalar HxEt = eps * ( +Iw * g * Eyt + (Iw/k2) * gxx * Eyt - (Iw/k2) * gxy * Ext ) * mrfac;

  PetscScalar HyHr = (-eps/mu) * gz * Hyr * mrfac;
  PetscScalar HyHt = (-eps/mu) * gz * Hyt * mrfac;
  PetscScalar HyEr = eps * ( -Iw * g * Exr + (Iw/k2) * gyx * Eyr - (Iw/k2) * gyy * Exr ) * mrfac;
  PetscScalar HyEt = eps * ( -Iw * g * Ext + (Iw/k2) * gyx * Eyt - (Iw/k2) * gyy * Ext ) * mrfac;
  
  PetscScalar HzHr = (eps/mu) * (gx * Hxr + gy * Hyr) * mrfac;
  PetscScalar HzHt = (eps/mu) * (gx * Hxt + gy * Hyt) * mrfac;
  PetscScalar HzEr = eps * ( +(Iw/k2) * gzx * Eyr - (Iw/k2) * gzy * Exr ) * mrfac;
  PetscScalar HzEt = eps * ( +(Iw/k2) * gzx * Eyt - (Iw/k2) * gzy * Ext ) * mrfac;

  fval[0] =creal(ExEr);
  fval[1] =creal(ExEt);
  fval[2] =creal(ExHr);
  fval[3] =creal(ExHt);
  fval[4] =creal(EyEr);
  fval[5] =creal(EyEt);
  fval[6] =creal(EyHr);
  fval[7] =creal(EyHt);
  fval[8] =creal(EzEr);
  fval[9] =creal(EzEt);
  fval[10]=creal(EzHr);
  fval[11]=creal(EzHt);
  fval[12]=creal(HxEr);
  fval[13]=creal(HxEt);
  fval[14]=creal(HxHr);
  fval[15]=creal(HxHt);
  fval[16]=creal(HyEr);
  fval[17]=creal(HyEt);
  fval[18]=creal(HyHr);
  fval[19]=creal(HyHt);
  fval[20]=creal(HzEr);
  fval[21]=creal(HzEt);
  fval[22]=creal(HzHr);
  fval[23]=creal(HzHt);

  fval[24]=cimag(ExEr);
  fval[25]=cimag(ExEt);
  fval[26]=cimag(ExHr);
  fval[27]=cimag(ExHt);
  fval[28]=cimag(EyEr);
  fval[29]=cimag(EyEt);
  fval[30]=cimag(EyHr);
  fval[31]=cimag(EyHt);
  fval[32]=cimag(EzEr);
  fval[33]=cimag(EzEt);
  fval[34]=cimag(EzHr);
  fval[35]=cimag(EzHt);
  fval[36]=cimag(HxEr);
  fval[37]=cimag(HxEt);
  fval[38]=cimag(HxHr);
  fval[39]=cimag(HxHt);
  fval[40]=cimag(HyEr);
  fval[41]=cimag(HyEt);
  fval[42]=cimag(HyHr);
  fval[43]=cimag(HyHt);
  fval[44]=cimag(HzEr);
  fval[45]=cimag(HzEt);
  fval[46]=cimag(HzHr);
  fval[47]=cimag(HzHt);

  return 0;
  
}

void near2far(Vec Er, Vec Et, Vec Hr, Vec Ht,
	      PetscReal x, PetscReal y, PetscReal z, PetscReal z0,
	      PetscInt nr_mtr, PetscReal dr,
	      PetscInt num_m, PetscInt *mlist,
	      PetscReal omega, PetscReal mu, PetscReal eps,
	      PetscScalar *farfield)
{

  PetscReal params[10]={x,y,z,z0, 0, dr, 0, omega,mu,eps};

  Vec fEH[24];
  PetscScalar *_fEH[24];
  for(PetscInt i=0;i<24;i++){
    VecDuplicate(Er,&fEH[i]);
    VecGetArray(fEH[i],&(_fEH[i]));
  }
    
  PetscInt ns,ne;
  VecGetOwnershipRange(Er,&ns,&ne);

  PetscReal theta0=0;
  PetscReal theta1=2*M_PI;
  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ir = (k=i)%nr_mtr;
    PetscInt im = (k/=nr_mtr)%num_m;
    PetscInt m = mlist[im];
    params[4] = (PetscReal)ir;
    params[6] = (PetscReal)m;

    PetscReal err[48];
    PetscReal EH[48];

    hcubature(48, fE, params,
	      1, &theta0, &theta1,
	      maxeval_green, abserr_green, relerr_green,
	      ERROR_L2,
	      EH, err);
    
    PetscInt j = i-ns;
    for(PetscInt ii=0;ii<24;ii++)
      _fEH[ii][j] = EH[ii] + PETSC_i * EH[ii+24];
    
  }

  for(PetscInt i=0;i<24;i++)
    VecRestoreArray(fEH[i],&(_fEH[i]));

  PetscScalar tmp1,tmp2,tmp3,tmp4;

  for(PetscInt i=0;i<6;i++){
    VecTDot(fEH[4*i+0],Er,&tmp1);
    VecTDot(fEH[4*i+1],Et,&tmp2);
    VecTDot(fEH[4*i+2],Hr,&tmp3);
    VecTDot(fEH[4*i+3],Ht,&tmp4);
    farfield[i] = tmp1+tmp2+tmp3+tmp4;
  }

  for(PetscInt i=0;i<24;i++)
    VecDestroy(&(fEH[i]));
  
}

