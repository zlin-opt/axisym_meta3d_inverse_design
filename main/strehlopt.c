#include "petsc.h"
#include "petscsys.h"
#include "nlopt.h"
#include <assert.h>
#include "libCYLOPT.h"

#define print_idspecs 0
#define print_idsims 0
#define print_mlist 0
#define print_tmpsolves 0
#define print_idsolves 0
#define verbose 0

#define MAXNUM_M 1000
#define MAX_IT 15

PetscInt count=0;

PetscInt get_ispec(PetscInt **idset_specs, PetscInt nspecs, PetscInt ifreq, PetscInt iangle, PetscInt ipol){

  PetscInt ispec,match=0;
  for(ispec=0;ispec<nspecs;ispec++){
    if( idset_specs[ispec][0]==ifreq && idset_specs[ispec][1]==iangle && idset_specs[ispec][2]==ipol ){
      match=1;
      break;
    }
  }
      
  if(match==0)
    return -1;
  else
    return ispec;
  
}

PetscInt get_im(PetscInt *mlist, PetscInt num_m, PetscInt m){

  PetscInt im,match=0;
  for(im=0;im<num_m;im++){
    if( mlist[im]==m ){
      match=1;
      break;
    }
  }

  if(match==0)
    return -1;
  else
    return im;

}

PetscInt get_isim(PetscInt **idset_sims, PetscInt nsims, PetscInt ispec, PetscInt icell, PetscInt im){

  PetscInt isim,match=0;
  for(isim=0;isim<nsims;isim++){
    if( idset_sims[isim][0]==ispec && idset_sims[isim][1]==icell && idset_sims[isim][2]==im ){
      match=1;
      break;
    }
  }

  if(match==0)
    return -1;
  else
    return isim;

}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);

  PetscInt nget;

  //Get the specifications: frequencies, angles and polarizations
  PetscInt nfreqs,nangles;
  getint("-nfreqs",&nfreqs,3);
  getint("-nangles",&nangles,3);
  PetscInt npols[nangles];
  nget=nangles;
  getintarray("-npols",npols,&nget,1);
  PetscInt nspecs=0;
  for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++)
    for(PetscInt iangle=0;iangle<nangles;iangle++)
      for(PetscInt ipol=0;ipol<npols[iangle];ipol++)
	nspecs++;
  PetscInt **idset_specs = (PetscInt **)malloc(nspecs*sizeof(PetscInt *));
  for(PetscInt i=0;i<nspecs;i++)
    idset_specs[i] = (PetscInt *)malloc(3*sizeof(PetscInt));
  PetscInt ispec=0;
  for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){
    for(PetscInt iangle=0;iangle<nangles;iangle++){
      for(PetscInt ipol=0;ipol<npols[iangle];ipol++){
	idset_specs[ispec][0]=ifreq;
	idset_specs[ispec][1]=iangle;
	idset_specs[ispec][2]=ipol;
	ispec++;
      }
    }
  }
  PetscReal freqs[nfreqs], angles[nangles];
  nget=nfreqs;
  getrealarray("-freqs",freqs,&nget,1.0);
  nget=nangles;
  getrealarray("-angles",angles,&nget,0.0);
  PetscPrintf(PETSC_COMM_WORLD,"NOTE: the total number of specs is %d. \n",nspecs);
  
  //DEBUG start
  if(print_idspecs)
    for(PetscInt ispec=0;ispec<nspecs;ispec++)
      PetscPrintf(PETSC_COMM_WORLD,"idset_specs[%d][freq,angle,pol] : %d, %d, %d \n",
		  ispec,
		  idset_specs[ispec][0],
		  idset_specs[ispec][1],
		  idset_specs[ispec][2]);
  //DEBUG end

  //Get nr, pr, pmlr, dr, ncells;
  PetscInt nr, pr, pmlr, ncells;
  getint("-nr",&nr,200);
  getint("-pr",&pr,50);
  getint("-pmlr",&pmlr,25);
  getint("-ncells",&ncells,5);
  PetscReal dr;
  getreal("-dr",&dr,0.04);

  //Get vertical layers information
  PetscInt nlayers_total,nlayers_active;
  getint("-nlayers_total", &nlayers_total, 3);
  getint("-nlayers_active",&nlayers_active,1);
  PetscInt thickness[nlayers_total];
  PetscInt id_active_layers[nlayers_active];
  nget=nlayers_total;
  getintarray("-thickness",thickness,&nget,10);
  nget=nlayers_active;
  getintarray("-id_active_layers",id_active_layers,&nget,0);
  PetscInt nz=integer_sum(thickness,0,nlayers_total);
  PetscPrintf(PETSC_COMM_WORLD,"NOTE: nz = %d \n",nz);
  PetscReal dz;
  PetscInt pmlz;
  getreal("-dz",&dz,dr);
  getint("-pmlz",&pmlz,pmlr);
  //Construct mzo and izo (starting z-coordinates for the active/all layers) 
  PetscInt mz[nlayers_active],mzo[nlayers_active];
  for(PetscInt i=0;i<nlayers_active;i++){
    mz[i]=thickness[id_active_layers[i]];
    mzo[i]=integer_sum(thickness,0,id_active_layers[i]);
  }
  PetscInt izo[nlayers_total];
  for(PetscInt i=0;i<nlayers_total;i++)
    izo[i]=integer_sum(thickness,0,i);
  PetscInt mzsum=integer_sum(mz,0,nlayers_active);

  
  //Get the epsilon information
  PetscReal epsbkg[nfreqs][2*nlayers_total];
  PetscReal epsfeg[nfreqs][2*nlayers_active];
  PetscScalar bkg_eps[nfreqs][nlayers_total];
  PetscScalar feg_eps[nfreqs][nlayers_active];
  PetscScalar diff_eps[nfreqs][nlayers_active];
  for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){
    
    char flag[PETSC_MAX_PATH_LEN];
    sprintf(flag,"-ifreq%d_epsbkg",ifreq);
    nget=2*nlayers_total;
    getrealarray(flag,epsbkg[ifreq],&nget,1);
    sprintf(flag,"-ifreq%d_epsfeg",ifreq);
    nget=2*nlayers_active;
    getrealarray(flag,epsfeg[ifreq],&nget,1);

    for(PetscInt j=0;j<nlayers_total;j++)
      bkg_eps[ifreq] [j]=epsbkg[ifreq][j]+PETSC_i*epsbkg[ifreq][j+nlayers_total];
    for(PetscInt j=0;j<nlayers_active;j++){
      feg_eps[ifreq] [j]=epsfeg[ifreq][j]+PETSC_i*epsfeg[ifreq][j+nlayers_active];
      diff_eps[ifreq][j]=feg_eps[ifreq][j]-bkg_eps[ifreq][id_active_layers[j]];
    }
    
  }

  //Determine mlist for each cell (picking up the Jacobi-Angers coefficients > cutoff)
  PetscReal cutoff;
  getreal("-cutoff",&cutoff,0.001);
  PetscInt **num_m = (PetscInt **)malloc(nspecs*sizeof(PetscInt*));
  PetscInt ***mlist = (PetscInt ***)malloc(nspecs*sizeof(PetscInt**));
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    num_m[ispec]=(PetscInt *)malloc(ncells*sizeof(PetscInt));
    mlist[ispec]=(PetscInt **)malloc(ncells*sizeof(PetscInt*));
  }
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    for(PetscInt icell=0;icell<ncells;icell++){

      PetscInt ifreq=idset_specs[ispec][0];
      PetscInt iangle=idset_specs[ispec][1];
      
      PetscReal rmin=icell*nr*dr;
      PetscReal rmax=(icell+1)*nr*dr;
      PetscReal omega=2*M_PI*freqs[ifreq];
      PetscReal nsub=sqrt(creal(bkg_eps[ifreq][0]));
      PetscReal k = nsub*omega;
      PetscReal alpha = angles[iangle]*M_PI/180;
      PetscReal phi = 0;
      m_select_nonneg(MAXNUM_M, rmin,rmax, k, phi,alpha, cutoff, &num_m[ispec][icell], &(mlist[ispec][icell]));
      MPI_Barrier(PETSC_COMM_WORLD);

      //DEBUG start
      if(print_mlist){
	PetscPrintf(PETSC_COMM_WORLD,"\nNOTE: specID %d cell %d num_m %d:  ",ispec,icell,num_m[ispec][icell]);
	for(PetscInt im=0;im<num_m[ispec][icell];im++)
	  PetscPrintf(PETSC_COMM_WORLD,"%d ",mlist[ispec][icell][im]);
	PetscPrintf(PETSC_COMM_WORLD,"\n");
	PetscPrintf(PETSC_COMM_WORLD,"NOTE: total number of azimuthal modes for spec %d is %d \n",ispec,integer_sum(num_m[ispec],0,ncells));
      }
      //DEBUG end

    }
  }
  MPI_Barrier(PETSC_COMM_WORLD);

  //Indexing the simulations
  PetscInt nsims=0;
  for(PetscInt ispec=0;ispec<nspecs;ispec++)
    for(PetscInt icell=0;icell<ncells;icell++)
      for(PetscInt im=0;im<num_m[ispec][icell];im++)
	nsims++;
  PetscInt **idset_sims=(PetscInt **)malloc(nsims*sizeof(PetscInt *));
  for(PetscInt i=0;i<nsims;i++)
    idset_sims[i]=(PetscInt *)malloc(3*sizeof(PetscInt));
  PetscInt isim=0;
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    for(PetscInt icell=0;icell<ncells;icell++){
      for(PetscInt im=0;im<num_m[ispec][icell];im++){
	idset_sims[isim][0]=ispec;
	idset_sims[isim][1]=icell;
	idset_sims[isim][2]=im;
	isim++;
      }
    }
  }
  PetscPrintf(PETSC_COMM_WORLD,"NOTE: the total number of simulations is %d. \n",nsims);

  //DEBUG start
  if(print_idsims)
    for(PetscInt isim=0;isim<nsims;isim++)
      PetscPrintf(PETSC_COMM_WORLD,"idset_sims[%d][spec,cell,m] : %d, %d, %d \n",
		  isim,
		  idset_sims[isim][0],
		  idset_sims[isim][1],
		  idset_sims[isim][2]);
  //DEBUG end

  PetscInt tmp[nsims][4];
  for(PetscInt i=0;i<nsims;i++)
    tmp[i][0]=-10, tmp[i][1]=-10, tmp[i][2]=-10, tmp[i][3]=0;

  PetscInt nsolves=0;
  for(PetscInt isim=0;isim<nsims;isim++){
    PetscInt ispec = idset_sims[isim][0];
    PetscInt ifreq = idset_specs[ispec][0];
    //PetscInt iangle = idset_specs[ispec][1];
    //PetscInt ipol = idset_specs[ispec][2];
    PetscInt icell = idset_sims[isim][1];
    PetscInt im = idset_sims[isim][2];
    PetscInt m = mlist[ispec][icell][im];
    
    PetscInt isolve, match=0;
    for(isolve=0;isolve<nsolves;isolve++){
      if(tmp[isolve][0]==ifreq && tmp[isolve][1]==icell && tmp[isolve][2]==m){
	match=1;
	break;
      }
    }

    if(match==0){
      tmp[nsolves][0]=ifreq, tmp[nsolves][1]=icell, tmp[nsolves][2]=m;
      tmp[nsolves][3]+=1;
      nsolves+=1;
    }
    if(match==1){
      tmp[isolve][3]+=1;
    }
    
  }
  PetscPrintf(PETSC_COMM_WORLD,"NOTE: total number of direct solves = %d \n",nsolves);
  
  //DEBUG start
  PetscInt totsolves=0;
  if(print_tmpsolves){
    for(PetscInt isolve=0;isolve<nsolves;isolve++){
      PetscPrintf(PETSC_COMM_WORLD,"tmp_solves[%d][ifreq,icell,m,msolves] : %d, %d, %d, %d\n",
		  isolve,
		  tmp[isolve][0],
		  tmp[isolve][1],
		  tmp[isolve][2],
		  tmp[isolve][3]);
      totsolves+=tmp[isolve][3];
    }
    PetscPrintf(PETSC_COMM_WORLD,"total number of solves = %d \n",totsolves);
  }
  //DEBUG end

  PetscInt ***idset_solves=(PetscInt ***)malloc(nsolves*sizeof(PetscInt **));
  for(PetscInt isolve=0;isolve<nsolves;isolve++){
    idset_solves[isolve]=(PetscInt **)malloc(3*sizeof(PetscInt *));
    idset_solves[isolve][0]=(PetscInt *)malloc(4*sizeof(PetscInt));
    idset_solves[isolve][1]=(PetscInt *)malloc(tmp[isolve][3]*sizeof(PetscInt));
    idset_solves[isolve][2]=(PetscInt *)malloc(tmp[isolve][3]*sizeof(PetscInt));
  }

  for(PetscInt isolve=0;isolve<nsolves;isolve++){

    PetscInt ifreq=tmp[isolve][0];
    PetscInt icell=tmp[isolve][1];
    PetscInt m=tmp[isolve][2];
    PetscInt msolves=tmp[isolve][3];
    idset_solves[isolve][0][0]=msolves;
    idset_solves[isolve][0][1]=ifreq;
    idset_solves[isolve][0][2]=icell;
    idset_solves[isolve][0][3]=m;

    PetscInt jsolve=0;
    for(PetscInt iangle=0;iangle<nangles;iangle++){
      for(PetscInt ipol=0;ipol<npols[iangle];ipol++){
	PetscInt ispec=get_ispec(idset_specs,nspecs,ifreq,iangle,ipol);
	if(ispec>=0){
	  PetscInt im = get_im(mlist[ispec][icell],num_m[ispec][icell],m);
	  if(im>=0){
	    PetscInt isim=get_isim(idset_sims,nsims,ispec,icell,im);
	    if(isim>=0){
	      idset_solves[isolve][1][jsolve]=ispec;
	      idset_solves[isolve][2][jsolve]=isim;
	      jsolve+=1;
	    }
	  }
	}
      }
    }

  }

  //DEBUG start
  if(print_idsolves){
    for(PetscInt isolve=0;isolve<nsolves;isolve++){
      PetscInt msolves=idset_solves[isolve][0][0];
      PetscPrintf(PETSC_COMM_WORLD,"idset_solves[%d] %d (ispec,isim):   ", isolve, msolves);
      for(PetscInt jsolve=0;jsolve<msolves;jsolve++)
	PetscPrintf(PETSC_COMM_WORLD,"(%d,%d) ", idset_solves[isolve][1][jsolve], idset_solves[isolve][2][jsolve]);
      PetscPrintf(PETSC_COMM_WORLD,"\n");
    }
  }

  //Get filter info, focal length, print_at and current amplitude; also angle / spatial compress factors for imaging
  PetscReal filter_radius,filter_sigma,filter_beta;
  getreal("-filter_radius",&filter_radius,3);
  getreal("-filter_sigma",&filter_sigma,10);
  getreal("-filter_beta",&filter_beta,60);
  PetscInt zfixed;
  getint("-zfixed",&zfixed,0);
  PetscReal foclen;
  getreal("-focal_length",&foclen,100);
  PetscReal virtual_foclen;
  getreal("-virtual_focal_length",&virtual_foclen,foclen);
  PetscInt print_at;
  getint("-print_at",&print_at,1);
  PetscReal angle_compress_factor, spatial_compress_factor;
  getreal("-angle_compress_factor",&angle_compress_factor,1.0);
  getreal("-spatial_compress_factor",&spatial_compress_factor,1.0);
  PetscInt iz_src,iz_mtr;
  getint("-iz_src",&iz_src,pmlz+5);
  getint("-iz_mtr",&iz_mtr,nz-pmlz-5);
  PetscReal maxstrehl[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){

    char flag[PETSC_MAX_PATH_LEN];
    sprintf(flag,"-ispec%d_maxstrehl",ispec);
    getreal(flag,&(maxstrehl[ispec]),1.0);

  }
  
  //Handle the MPI splitting
  PetscInt nprocs_total;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_total);
  PetscInt nprocs_per_solve;
  PetscInt nsolves_per_comm;
  PetscInt ncomms;
  PetscInt nprocs_error=-10;
  for(nsolves_per_comm=1;nsolves_per_comm<=nsolves;nsolves_per_comm++){
    for(nprocs_per_solve=1;nprocs_per_solve<=nprocs_total;nprocs_per_solve++){
      nprocs_error = nprocs_per_solve * nsolves - nsolves_per_comm * nprocs_total;
      ncomms = nsolves/nsolves_per_comm;
      if(nprocs_error == 0)
	break;
    }
    if(nprocs_error == 0)
      break;
  }
  if(nprocs_error != 0)
    SETERRQ(PETSC_COMM_WORLD,1,"The total number of processors is not consistent with the given number of solves.");
  else
    PetscPrintf(PETSC_COMM_WORLD,"***NOTE: nprocs_per_solve ( %d ) x nsolves ( %d ) = nsolves_per_comm ( %d ) x nprocs_total ( %d ), ncomms= %d \n",
		nprocs_per_solve, nsolves,
		nsolves_per_comm, nprocs_total,
		ncomms);
  PetscInt nprocs_per_comm=nprocs_per_solve;

  PetscInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm subcomm;
  PetscInt colour = rank/nprocs_per_comm;
  MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);


  //###################################################################################
  //BEGIN building up the vectors and matrices

  PetscReal fspotx[nspecs], fspoty[nspecs], fspotz[nspecs];
  ispec=0;
  for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){
    for(PetscInt iangle=0;iangle<nangles;iangle++){
      for(PetscInt ipol=0;ipol<npols[iangle];ipol++){

	PetscReal alpha=angles[iangle]*M_PI/180;
	PetscReal n0=sqrt(creal(bkg_eps[0][0]));
	PetscReal output_angle=asin(n0*sin(alpha));
	fspotx[ispec]=virtual_foclen*tan(output_angle*angle_compress_factor)*spatial_compress_factor;
	fspoty[ispec]=0;
	fspotz[ispec]=foclen;
	PetscPrintf(PETSC_COMM_WORLD,"INFO: the focal spot for freq %g, input angle %g, output angle %g and polarization %d [specid %d] is (%g,%g,%g) \n",
		    freqs[ifreq], angles[iangle], output_angle*180/M_PI, ipol,
		    ispec,
		    fspotx[ispec],fspoty[ispec],fspotz[ispec]);
	
	ispec++;
      }
    }
  }  
  
  //Construct smear + odm matrix
  Mat Q1,Q2,W;
  PetscInt mrows_per_cell[ncells],cell_start[ncells];
  density_filter(PETSC_COMM_WORLD, &Q1, nr,ncells,nlayers_active, filter_radius,filter_sigma, 1);
  create_ovmat(PETSC_COMM_WORLD, &Q2, mrows_per_cell, cell_start, nr,pr,ncells,nlayers_active, pmlr,pmlr, mz, 1);
  MatMatMult(Q2,Q1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&W);
  MatDestroy(&Q1);
  MatDestroy(&Q2);  

  params_ params;
  params.nfreqs=nfreqs;
  params.nangles=nangles;
  params.npols=npols;
  params.freqs=freqs;
  params.angles=angles;
  params.num_m=num_m;
  params.mlist=mlist;
  params.nspecs=nspecs;
  params.nsims=nsims;
  params.nsolves=nsolves;
  params.idset_specs=idset_specs;
  params.idset_sims=idset_sims;
  params.idset_solves=idset_solves;

  params.nr=nr;
  params.pr=pr;
  params.pmlr=pmlr;
  params.ncells=ncells;
  params.nz=nz;
  params.pmlz=pmlz;
  params.dr=dr;
  params.dz=dz;
  params.iz_src=iz_src;
  params.iz_mtr=iz_mtr;
  params.nlayers_active=nlayers_active;
  params.mz=mz;
  params.mzsum=mzsum;

  params.W=W;
  params.filter_beta=filter_beta;
  params.zfixed=zfixed;

  params.subcomm=subcomm;
  params.nsolves_per_comm=nsolves_per_comm;
  params.ncomms=ncomms;
  params.colour=colour;

  Mat A[nsolves_per_comm];
  Mat DDe[nsolves_per_comm];
  Vec epsDiff[nsolves_per_comm];
  Vec epsBkg[nsolves_per_comm];
  KSP ksp[nsolves_per_comm];
  PetscInt its[nsolves_per_comm];
  PetscInt maxit=15;
  Mat Qp[nsolves_per_comm];
  Mat Qp_minus[nsolves_per_comm];
  Vec gex[2*nangles*nsolves_per_comm];
  Vec gey[2*nangles*nsolves_per_comm];
  Vec ghx[2*nangles*nsolves_per_comm];
  Vec ghy[2*nangles*nsolves_per_comm];
  Vec gex_minus[2*nangles*nsolves_per_comm];
  Vec gey_minus[2*nangles*nsolves_per_comm];
  Vec ghx_minus[2*nangles*nsolves_per_comm];
  Vec ghy_minus[2*nangles*nsolves_per_comm];
  Vec sigmahat[2*nangles*nsolves_per_comm];
  params.Jr=(PetscScalar ***)malloc(nsolves_per_comm*sizeof(PetscScalar **));
  params.Jt=(PetscScalar ***)malloc(nsolves_per_comm*sizeof(PetscScalar **));
  for(PetscInt isolve_per_comm=0;isolve_per_comm<nsolves_per_comm;isolve_per_comm++){

    PetscInt isolve  = isolve_per_comm + nsolves_per_comm * colour;
    PetscInt msolves = idset_solves[isolve][0][0];
    PetscInt ifreq   = idset_solves[isolve][0][1];
    PetscInt icell   = idset_solves[isolve][0][2];
    PetscInt m       = idset_solves[isolve][0][3];
    PetscInt nr_cell = (icell==0) ? nr+pr : pr+nr+pr;
    PetscReal omega=2*M_PI*freqs[ifreq];
    
    create_Ainterp(subcomm, &(A[isolve_per_comm]),
		   nr_cell, nz,
		   0, nr_cell, mzo, mz, 0,
		   nlayers_active);

    create_DDe(subcomm, &(DDe[isolve_per_comm]),
	       nr_cell, nz,
	       (icell==0) ? 0 : pmlr,pmlr, pmlz,pmlz,
	       dr,dz, (icell==0) ? 0 : (icell*nr - pr)*dr,
	       omega,
	       m);

    setupKSPDirect(subcomm, &ksp[isolve_per_comm], maxit);
    its[isolve_per_comm]=100;

    VecCreateMPI(subcomm,PETSC_DECIDE,nr_cell*nz*3,&epsBkg[isolve_per_comm]);
    VecCreateMPI(subcomm,PETSC_DECIDE,nr_cell*nz*3,&epsDiff[isolve_per_comm]);
    setlayer_eps(epsBkg[isolve_per_comm],  nr_cell,nz, nlayers_total,  izo, thickness, bkg_eps[ifreq]);
    setlayer_eps(epsDiff[isolve_per_comm], nr_cell,nz, nlayers_active, mzo, mz,        diff_eps[ifreq]);

    create_Qp(subcomm, &Qp[isolve_per_comm],
	      nr_cell, nz,
	      (icell==0) ? 0 : pmlr, pmlr, pmlz,pmlz,
	      dr,dz,
	      omega,
	      m,
	      (icell==0) ? 0 : pr, nr, icell*nr, iz_mtr);   //starts from pr for non-center cells
    create_Qp(subcomm, &Qp_minus[isolve_per_comm],
	      nr_cell, nz,
	      (icell==0) ? 0 : pmlr, pmlr, pmlz,pmlz,
	      dr,dz,
	      omega,
	      -m,
	      (icell==0) ? 0 : pr, nr, icell*nr, iz_mtr);   //starts from pr for non-center cells
    
    params.Jr[isolve_per_comm]=(PetscScalar **)malloc(msolves*sizeof(PetscScalar *));
    params.Jt[isolve_per_comm]=(PetscScalar **)malloc(msolves*sizeof(PetscScalar *));
    for(PetscInt jsolve=0;jsolve<msolves;jsolve++){
      
      PetscInt ispec = idset_solves[isolve][1][jsolve];
      PetscInt isim = idset_solves[isolve][2][jsolve];
      PetscInt iangle = idset_specs[ispec][1];
      PetscInt ipol   = idset_specs[ispec][2];      
      
      PetscPrintf(subcomm,"Constructing params for [ sim %04d, spec %04d, solve %04d ] for (( pol %d, cell %04d, m %04d, freq %0.3g, angle %0.3g )) in comm %04d \n",
		  isim,ispec,isolve,
		  ipol,icell,m,freqs[ifreq],angles[iangle],
		  colour);
    
      PetscReal nsub=sqrt(creal(bkg_eps[ifreq][0]));
      PetscReal k=nsub*omega;
      PetscReal alpha=angles[iangle]*M_PI/180;
      PetscReal phi=0;
      PetscScalar Ax=0,Ay=0;
      if(ipol==0)
	Ax=1,Ay=0;
      else
	Ax=0,Ay=1;
      params.Jr[isolve_per_comm][jsolve]=(PetscScalar *)malloc(nr_cell*sizeof(PetscScalar));
      params.Jt[isolve_per_comm][jsolve]=(PetscScalar *)malloc(nr_cell*sizeof(PetscScalar));
      Jm(params.Jr[isolve_per_comm][jsolve],params.Jt[isolve_per_comm][jsolve],
	 (icell==0) ? 0 : icell*nr - pr, nr_cell,    //J array starts from zeroth index 
	 dr,dz,
	 m,
	 k, alpha, phi,
	 Ax,Ay);
      
      VecDuplicate(epsBkg[isolve_per_comm],&gex[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&gey[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&ghx[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&ghy[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&gex_minus[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&gey_minus[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&ghx_minus[jsolve + 2*nangles * isolve_per_comm]);
      VecDuplicate(epsBkg[isolve_per_comm],&ghy_minus[jsolve + 2*nangles * isolve_per_comm]);
      gforms(subcomm, gex[jsolve + 2*nangles * isolve_per_comm], gey[jsolve + 2*nangles * isolve_per_comm], ghx[jsolve + 2*nangles * isolve_per_comm], ghy[jsolve + 2*nangles * isolve_per_comm],
	     nr_cell, nz,
	     (icell==0) ? 0 : pmlr, pmlr, pmlz,pmlz,
	     dr,dz,
	     omega,
	     m,
	     (icell==0) ? 0 : pr, nr, icell*nr, iz_mtr,
	     fspotx[ispec], fspoty[ispec], fspotz[ispec], 0,
	     1,1);
      gforms(subcomm, gex_minus[jsolve + 2*nangles * isolve_per_comm], gey_minus[jsolve + 2*nangles * isolve_per_comm], ghx_minus[jsolve + 2*nangles * isolve_per_comm], ghy_minus[jsolve + 2*nangles * isolve_per_comm],
	     nr_cell, nz,
	     (icell==0) ? 0 : pmlr, pmlr, pmlz,pmlz,
	     dr,dz,
	     omega,
	     -m,
	     (icell==0) ? 0 : pr, nr, icell*nr, iz_mtr,
	     fspotx[ispec], fspoty[ispec], fspotz[ispec], 0,
	     1,1);

      VecDuplicate(epsBkg[isolve_per_comm],&sigmahat[jsolve + 2*nangles * isolve_per_comm]);
      make_sigmahat(sigmahat[jsolve + 2*nangles * isolve_per_comm], nr_cell, nz, ipol);

    }
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    
  }
  MPI_Barrier(subcomm);
  MPI_Barrier(PETSC_COMM_WORLD);

  params.epsDiff=epsDiff;
  params.epsBkg=epsBkg;
  params.A=A;
  params.DDe=DDe;
  params.Qp=Qp;
  params.Qp_minus=Qp_minus;
  params.gex=gex;
  params.gey=gey;
  params.ghx=ghx;
  params.ghy=ghy;
  params.gex_minus=gex_minus;
  params.gey_minus=gey_minus;
  params.ghx_minus=ghx_minus;
  params.ghy_minus=ghy_minus;
  params.sigmahat=sigmahat;
  params.ksp=ksp;
  params.its=its;
  params.maxit=maxit;
  PetscPrintf(PETSC_COMM_WORLD,"**********Params assigned.\n");

  params.airyfactor=(PetscReal *)malloc(nspecs*sizeof(PetscReal));
  for(PetscInt ispec=0;ispec<nspecs;ispec++){

    PetscInt ifreq=idset_specs[ispec][0];
    PetscReal freq=freqs[ifreq];
    PetscReal NA=sin(atan(nr*ncells*dr/foclen));
    params.airyfactor[ispec] = (4.0*M_PI/pow(6.46536,2)) * (1.0/pow(freq*NA,2)) * (1.0/maxstrehl[ispec]);

  }

  //Initialize the degrees of freedom
  PetscInt nhdof=nr*ncells*nlayers_active;
  PetscReal *hdof = (PetscReal *)malloc(nhdof*sizeof(PetscReal));
  PetscBool flg;
  char strin[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-initial_filename",strin,PETSC_MAX_PATH_LEN-1,&flg);
  if(flg){
    PetscPrintf(PETSC_COMM_WORLD,"--initial_filename is %s \n",strin);
    readfromfile(strin, hdof, REAL, nhdof);
  }else{
    PetscReal autoinit;
    getreal("-autoinit",&autoinit,0.5);
    PetscPrintf(PETSC_COMM_WORLD,"***NOTE: NO initial filename given. Auto-initialized to %g\n",autoinit);
    for(PetscInt i=0;i<nhdof;i++)
      hdof[i] = autoinit;
  }

  PetscInt Job;
  getint("-Job",&Job,1);
  
  if(Job==-1){

    PetscInt idof;
    getint("-idof",&idof,nhdof/2);
    PetscReal ss[3];
    nget=3;
    getrealarray("-s0,s1,ds",ss,&nget,0);
    PetscReal s0=ss[0], s1=ss[1], ds=ss[2];
    
    PetscInt ndof=nhdof+1;
    PetscReal *result=(PetscReal *)malloc(nspecs*sizeof(PetscReal));
    PetscReal *dof=(PetscReal *)malloc(ndof*sizeof(PetscReal));
    PetscReal *grad=(PetscReal *)malloc(nspecs*ndof*sizeof(PetscReal));
    for(PetscInt i=0;i<nhdof;i++)
      dof[i]=hdof[i];
    dof[nhdof]=0;

    for(PetscReal s=s0;s<s1;s+=ds){
      dof[idof]=s;
      strehl((unsigned) nspecs, result,
	     (unsigned) ndof, dof, 
	     grad,
	     &params);
      for(PetscInt specID=0;specID<nspecs;specID++){
	PetscReal objval = result[specID];
	PetscReal gradval = grad[idof + ndof*specID];
	PetscPrintf(PETSC_COMM_WORLD,"spec%d objval: %g %0.16g %0.16g \n", specID, s, objval, gradval);
      }
    }

    free(result);
    free(dof);
    free(grad);
  }

  if(Job==1){

    PetscInt ndof = nhdof+1;

    PetscInt algouter, alginner, algmaxeval;
    getint("-algouter",&algouter,24);
    getint("-alginner",&alginner,24);
    getint("-algmaxeval",&algmaxeval,500);

    PetscReal epi_t;
    getreal("-epi_t",&epi_t,0);
    
    PetscReal *lb=(PetscReal *)malloc(ndof*sizeof(PetscReal));
    PetscReal *ub=(PetscReal *)malloc(ndof*sizeof(PetscReal));
    PetscReal *dof=(PetscReal *)malloc(ndof*sizeof(PetscReal));
    for(PetscInt i=0;i<nhdof;i++){
      lb[i]=0.0;
      ub[i]=1.0;
      dof[i]=hdof[i];
    }
    lb[nhdof]=0.0;
    ub[nhdof]=1.0;
    dof[nhdof]=epi_t;
    MPI_Barrier(PETSC_COMM_WORLD);
    
    nlopt_opt opt;
    nlopt_opt local_opt;
    opt = nlopt_create((nlopt_algorithm)algouter, ndof);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,algmaxeval);
    nlopt_set_maxtime(opt,100000000);

    if(alginner){
      local_opt=nlopt_create(alginner, ndof);
      nlopt_set_ftol_rel(local_opt, 1e-6);
      nlopt_set_maxeval(local_opt,10000);
      nlopt_set_local_optimizer(opt,local_opt);
    }

    PetscReal *mctol = (PetscReal *)malloc(nspecs*sizeof(PetscReal));
    for(PetscInt i=0;i<nspecs;i++)
      mctol[i]=1e-8;
    nlopt_add_inequality_mconstraint(opt, (unsigned) nspecs, (nlopt_mfunc) strehl, &params, mctol);
    
    nlopt_set_max_objective(opt,(nlopt_func)dummy_obj,&print_at);

    PetscReal maxf;
    nlopt_result result=nlopt_optimize(opt,dof,&maxf);
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: nlopt_result: %d \n",result);
    
    nlopt_destroy(opt);
    if(alginner) nlopt_destroy(local_opt);
    
    free(lb);
    free(ub);
    free(dof);
    free(mctol);

  }

  if(Job==0){

    PetscInt specID;
    getint("-specID",&specID,0);
    char output_prefix[PETSC_MAX_PATH_LEN];
    getstr("-output_prefix",output_prefix,"Job0");
    PetscInt printfields;
    getint("-printfields", &printfields, 0);
    
    PetscInt ifreq=idset_specs[specID][0];
    PetscInt iangle=idset_specs[specID][1];
    PetscInt ipol=idset_specs[specID][2];

    PetscInt mnum=2*num_m[specID][ncells-1]-1;
    if(mnum==1) mnum=2;
    PetscInt mlst[mnum];
    if(mnum==2){
      mlst[0]=-1, mlst[1]=1;
    }else{
      for(PetscInt i=0;i<mnum;i++){
	if(i<(mnum+1)/2)
	  mlst[i]=-mlist[specID][ncells-1][(mnum+1)/2 - i -1];
	else
	  mlst[i]= mlist[specID][ncells-1][i - (mnum+1)/2 +1];
      }
    }
      
    PetscInt nhdof1=nr*ncells*nlayers_active;
    PetscInt nhdof2=(nr*ncells+pr)*nlayers_active;
    PetscInt Nr = nr*ncells+pr;

    Mat Q;
    density_filter(PETSC_COMM_WORLD, &Q, nr,ncells,nlayers_active, filter_radius,filter_sigma, 1);
    PetscScalar *tmp1 = (PetscScalar *)malloc(nhdof1*sizeof(PetscScalar));
    PetscScalar *tmp2 = (PetscScalar *)malloc(nhdof2*sizeof(PetscScalar));
    matmult_arrays(Q, hdof,REAL, tmp1,SCAL, 0);
    for(PetscInt il=0;il<nlayers_active;il++)
      for(PetscInt ir=0;ir<Nr;ir++)
	if(ir<nr*ncells)
	  tmp2[ir+Nr*il]=tmp1[ir+nr*ncells*il];
	else
	  tmp2[ir+Nr*il]=0;
	

    PetscInt nedof = (nr*ncells+pr)*mzsum;
    PetscScalar *edof = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    varh_expand(tmp2, edof, Nr, 1, nlayers_active, mz, filter_beta, zfixed);

    Mat Afull;
    create_Ainterp(PETSC_COMM_WORLD, &Afull,
		   Nr, nz,
		   0, Nr, mzo, mz, 0,
		   nlayers_active);
    Vec u,v;
    MatCreateVecs(Afull,&u,&v);
    array2mpi(edof,SCAL, u);
    MatMult(Afull,u,v);

    Vec eps,epsBkg_full,epsDiff_full;
    VecDuplicate(v,&eps);
    VecDuplicate(v,&epsBkg_full);
    VecDuplicate(v,&epsDiff_full);
    setlayer_eps(epsDiff_full, Nr,nz, nlayers_active, mzo, mz,        diff_eps[ifreq]);
    setlayer_eps(epsBkg_full,  Nr,nz, nlayers_total,  izo, thickness, bkg_eps[ifreq]);
    VecPointwiseMult(eps,v,epsDiff_full);
    VecAXPY(eps,1.0,epsBkg_full);

    PetscInt eps_size;
    VecGetSize(eps,&eps_size);
    PetscScalar *_eps=(PetscScalar *)malloc(eps_size*sizeof(PetscScalar));
    mpi2array(eps,_eps,SCAL,eps_size);
    char eps_filename[PETSC_MAX_PATH_LEN];
    sprintf(eps_filename,"%s.epsilon.qrzc",output_prefix);
    writetofile(PETSC_COMM_WORLD,eps_filename,_eps,SCAL,eps_size);
    free(_eps);

    PetscReal omega=2*M_PI*freqs[ifreq];
    PetscReal nsub=sqrt(creal(bkg_eps[ifreq][0]));
    PetscReal k=nsub*omega;
    PetscReal alpha=angles[iangle]*M_PI/180;
    PetscReal phi=0;
    PetscScalar Ax=0,Ay=0;
    if(ipol==0)
      Ax=1,Ay=0;
    else
      Ax=0,Ay=1;
    
    PetscReal near_z0=0;
    PetscInt Nr_mtr;
    getint("-Nr_mtr",&Nr_mtr,nr*ncells);
    
    PetscScalar *_Er= (PetscScalar *)malloc(Nr_mtr*mnum*sizeof(PetscScalar));
    PetscScalar *_Et= (PetscScalar *)malloc(Nr_mtr*mnum*sizeof(PetscScalar));
    PetscScalar *_Hr= (PetscScalar *)malloc(Nr_mtr*mnum*sizeof(PetscScalar));
    PetscScalar *_Ht= (PetscScalar *)malloc(Nr_mtr*mnum*sizeof(PetscScalar));

    for(PetscInt im=0;im<mnum;im++){

      PetscInt m = mlst[im];
      PetscPrintf(PETSC_COMM_WORLD,"Started simulating m = %d\n",m);
      
      PetscScalar *Jmr = (PetscScalar *)malloc(Nr*sizeof(PetscScalar));
      PetscScalar *Jmt = (PetscScalar *)malloc(Nr*sizeof(PetscScalar));
      Jm(Jmr,Jmt,
	 0,Nr, dr,dz,
	 m,
	 k, alpha, phi,
	 Ax,Ay);

      Vec b;
      VecDuplicate(v,&b);
      VecSet(b,0);
      vecfill_zslice(b,Jmr,Jmt,Nr,nz,iz_src);
      VecScale(b,PETSC_i*omega);

      Vec mu;
      VecDuplicate(b,&mu);
      VecSet(mu,1.0);

      Vec _w2eps;
      VecDuplicate(b,&_w2eps);
      VecCopy(eps,_w2eps);
      VecScale(_w2eps,-omega*omega+PETSC_i*0.0);

      Vec x,xx;
      VecDuplicate(v,&x);
      VecDuplicate(v,&xx);

      Mat DDe;
      create_DDe(PETSC_COMM_WORLD, &DDe,
		 Nr, nz,
		 0, pmlr, pmlz, pmlz,
		 dr, dz, 0,
		 omega,
		 m);

      MatDiagonalSet(DDe,_w2eps,ADD_VALUES);

      KSP ksp;
      PetscInt maxit=15;
      PetscInt its=100;
      setupKSPDirect(PETSC_COMM_WORLD, &ksp, maxit);
      SolveMatrixDirect(PETSC_COMM_WORLD, ksp, DDe, b, xx, &its, maxit);
      KSPDestroy(&ksp);
      MPI_Barrier(PETSC_COMM_WORLD);

      Mat sE;
      syncE(PETSC_COMM_WORLD,&sE, Nr,nz);
      MatMult(sE,xx,x);

      Mat De;
      create_De(PETSC_COMM_WORLD, &De,
		Nr,nz,
		0,pmlr,pmlz,pmlz,
		dr,dz,0,
		omega,
		m);
      Vec yy,y;
      VecDuplicate(v,&y);
      VecDuplicate(v,&yy);
      MatMult(De,xx,yy);
      VecScale(yy,1/(PETSC_i*omega));
      VecPointwiseDivide(yy,yy,mu);
      Mat sH;
      syncH(PETSC_COMM_WORLD,&sH, Nr,nz);
      MatMult(sH,yy,y);

      PetscInt x_size;
      VecGetSize(x,&x_size);
      PetscScalar *_x=(PetscScalar *)malloc(x_size*sizeof(PetscScalar));
      mpi2array(x,_x,SCAL,x_size);
      PetscInt y_size=x_size;
      PetscScalar *_y=(PetscScalar *)malloc(y_size*sizeof(PetscScalar));
      mpi2array(y,_y,SCAL,y_size);

      for(PetscInt ir=0;ir<Nr_mtr;ir++){
	_Er[ir+Nr_mtr*im]=_x[ir+Nr*iz_mtr+Nr*nz*0];
	_Et[ir+Nr_mtr*im]=_x[ir+Nr*iz_mtr+Nr*nz*1];
	_Hr[ir+Nr_mtr*im]=_y[ir+Nr*iz_mtr+Nr*nz*0];
	_Ht[ir+Nr_mtr*im]=_y[ir+Nr*iz_mtr+Nr*nz*1];
      }

      if(printfields){

	char filename[PETSC_MAX_PATH_LEN];
	if(m>=0)
	  sprintf(filename,"%s.Efield.p%d.qrzc",output_prefix,abs(m));
	else
	  sprintf(filename,"%s.Efield.n%d.qrzc",output_prefix,abs(m));
	writetofile(PETSC_COMM_WORLD, filename, _x,SCAL, x_size);
	if(m>=0)
	  sprintf(filename,"%s.Hfield.p%d.qrzc",output_prefix,abs(m));
	else
	  sprintf(filename,"%s.Hfield.n%d.qrzc",output_prefix,abs(m));
	writetofile(PETSC_COMM_WORLD, filename, _y,SCAL, y_size);

      }

      VecDestroy(&b);
      VecDestroy(&mu);
      VecDestroy(&_w2eps);
      VecDestroy(&x);
      VecDestroy(&xx);
      VecDestroy(&y);
      VecDestroy(&yy);
      MatDestroy(&DDe);
      MatDestroy(&De);
      MatDestroy(&sE);
      MatDestroy(&sH);
      free(Jmr);
      free(Jmt);
      free(_x);
      free(_y);

      MPI_Barrier(PETSC_COMM_WORLD);

      PetscPrintf(PETSC_COMM_WORLD,"Finsished simulating m = %d\n",m);
      
    }
    
    PetscReal xcen_far,ycen_far,zcen_far;
    PetscInt nx_far,ny_far,nz_far;
    PetscReal dx_far,dy_far,dz_far;
    getreal("-xcen_far",&xcen_far,fspotx[specID]);
    getreal("-ycen_far",&ycen_far,fspoty[specID]);
    getreal("-zcen_far",&zcen_far,fspotz[specID]);
    getreal("-dx_far",&dx_far,0.2);
    getreal("-dy_far",&dy_far,0.2);
    getreal("-dz_far",&dz_far,0.01);
    getint("-nx_far",&nx_far,200);
    getint("-ny_far",&ny_far,200);
    getint("-nz_far",&nz_far,1);
    PetscInt nxyz=nx_far*ny_far*nz_far;

    Vec Er,Et,Hr,Ht;
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,Nr_mtr*mnum,&Er);
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,Nr_mtr*mnum,&Et);
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,Nr_mtr*mnum,&Hr);
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,Nr_mtr*mnum,&Ht);
    array2mpi(_Er,SCAL, Er);
    array2mpi(_Et,SCAL, Et);
    array2mpi(_Hr,SCAL, Hr);
    array2mpi(_Ht,SCAL, Ht);

    PetscScalar *fEx=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscScalar *fEy=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscScalar *fEz=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscScalar *fHx=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscScalar *fHy=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscScalar *fHz=(PetscScalar *)malloc(nx_far*ny_far*nz_far*sizeof(PetscScalar));
    PetscReal *fEE=(PetscReal *)malloc(nx_far*ny_far*nz_far*sizeof(PetscReal));
    PetscReal *fSz=(PetscReal *)malloc(nx_far*ny_far*nz_far*sizeof(PetscReal));
    for(PetscInt iz=0;iz<nz_far;iz++){
      for(PetscInt iy=0;iy<ny_far;iy++){
	for(PetscInt ix=0;ix<nx_far;ix++){
	  PetscReal x_far = xcen_far - nx_far*dx_far/2.0 + (ix+0.5)*dx_far;
	  PetscReal y_far = ycen_far - ny_far*dy_far/2.0 + (iy+0.5)*dy_far;
	  PetscReal z_far = zcen_far - nz_far*dz_far/2.0 + (iz+0.5)*dz_far;
	  PetscScalar fEH[6]={0,0,0,0,0,0};
	  near2far(Er, Et, Hr, Ht,
		   x_far, y_far, z_far, near_z0,
		   Nr_mtr, dr,
		   mnum, mlst,
		   omega,1,1,
		   fEH);
	  PetscInt ixyz=ix+nx_far*iy+nx_far*ny_far*iz;
	  fEx[ixyz]=fEH[0];
	  fEy[ixyz]=fEH[1];
	  fEz[ixyz]=fEH[2];
	  fHx[ixyz]=fEH[3];
	  fHy[ixyz]=fEH[4];
	  fHz[ixyz]=fEH[5];

	  fEE[ixyz]=pow(cabs(fEx[ixyz]),2) + pow(cabs(fEy[ixyz]),2) + pow(cabs(fEz[ixyz]),2);
	  fSz[ixyz]=creal( fEx[ixyz] * conj(fHy[ixyz]) - fEy[ixyz] * conj(fHx[ixyz]) );

	  PetscPrintf(PETSC_COMM_WORLD,"Farfield calculations %g percent done.\n",(PetscReal)ixyz*100/(PetscReal)nxyz);

	}
      }
    }

    PetscReal fEEmax=find_max(fEE,nxyz);
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: Peak far field intensity is %0.16g . \n",fEEmax);
    for(PetscInt i=0;i<nxyz;i++)
      fEE[i]=fEE[i]/fEEmax;

    PetscReal fSzmax=find_max(fSz,nxyz);
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: Peak far field Sz is %0.16g . \n",fSzmax);
    
    char filename[PETSC_MAX_PATH_LEN];
    sprintf(filename,"%s.Ex_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fEx,SCAL, nxyz);
    sprintf(filename,"%s.Ey_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fEy,SCAL, nxyz);
    sprintf(filename,"%s.Ez_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fEz,SCAL, nxyz);
    sprintf(filename,"%s.Hx_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fHx,SCAL, nxyz);
    sprintf(filename,"%s.Hy_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fHy,SCAL, nxyz);
    sprintf(filename,"%s.Hz_far.qxyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fHz,SCAL, nxyz);
    sprintf(filename,"%s.EE_far.xyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fEE,REAL, nxyz);
    sprintf(filename,"%s.Sz_far.xyz",output_prefix);
    writetofile(PETSC_COMM_WORLD, filename, fSz,REAL, nxyz);

    PetscReal trPz=0;
    for(PetscInt im=0;im<mnum;im++){
      for(PetscInt ir=0;ir<Nr_mtr;ir++){
	PetscReal r = (ir+0.5)*dr;
	PetscInt i = ir+Nr_mtr*im;
	trPz += creal( _Er[i]*conj(_Ht[i]) - _Et[i]*conj(_Hr[i]) ) * 2 * M_PI * r * dr;
      }
    }
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: Transmitted power above the lens is %0.16g \n",trPz);
    PetscReal Strehl_ratio = params.airyfactor[specID]*fSzmax/trPz;
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: calculated Strehl ratio is %0.16g \n",Strehl_ratio);
    
    PetscInt printeff;
    getint("-printeff",&printeff,0);
    if(printeff){

      PetscReal fwhm_rad;
      getreal("-fwhm_rad",&fwhm_rad,1);
      PetscReal NA = sin(atan(Nr_mtr*dr/foclen));
      PetscReal fwhm = 1/(2*freqs[ifreq]*NA);
      getreal("-fwhm",&fwhm,fwhm);
      PetscReal int_rad = fwhm*fwhm_rad;

      PetscReal fPz=0;
      for(PetscInt iy=0;iy<ny_far;iy++){
	for(PetscInt ix=0;ix<nx_far;ix++){
	  PetscReal x_far = (ix+0.5-nx_far/2.0)*dx_far;
	  PetscReal y_far = (iy+0.5-ny_far/2.0)*dy_far;
	  PetscReal r_far = sqrt(x_far*x_far + y_far*y_far);
	  if( r_far <= int_rad )
	    fPz += fSz[ix+nx_far*iy] * dx_far * dy_far;
	}
      }

      PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: Integrated flux within a radius of %g x FWHM is %0.16g \n",fwhm_rad,fPz);
      PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: Relative focusing efficiency is %0.16g \n", fPz/trPz);
      
    }

    free(fEx);
    free(fEy);
    free(fEz);
    free(fHx);
    free(fHy);
    free(fHz);
    free(fEE);
    free(fSz);
    
    MatDestroy(&Q);
    free(tmp1);
    free(tmp2);
    free(edof);
    MatDestroy(&Afull);
    VecDestroy(&u);
    VecDestroy(&v);
    VecDestroy(&eps);
    VecDestroy(&epsBkg_full);
    VecDestroy(&epsDiff_full);
    free(_Er);
    free(_Et);
    free(_Hr);
    free(_Ht);
    VecDestroy(&Er);
    VecDestroy(&Et);
    VecDestroy(&Hr);
    VecDestroy(&Ht);
    
  }

  MPI_Barrier(PETSC_COMM_WORLD);
  

  /*
  MatDestroy(&W);

  for(PetscInt i=0;i<nspecs;i++)
    free(idset_specs[i]);
  free(idset_specs);
  for(PetscInt i=0;i<nsims;i++)
    free(idset_sims[i]);
  free(idset_sims);
  MPI_Barrier(PETSC_COMM_WORLD);
  */

  PetscFinalize();

}
