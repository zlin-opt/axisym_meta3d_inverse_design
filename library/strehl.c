#include "strehl.h"

extern PetscInt count;

void strehl(unsigned nspecs, PetscReal *result,
	    unsigned ndof, const PetscReal *dof,
	    PetscReal *grad,
	    void *data)
{

  params_ *params = (params_ *)data;

  PetscInt nr = params->nr;
  PetscInt pr = params->pr;
  PetscInt ncells = params->ncells;
  PetscInt nlayers_active = params->nlayers_active;
  Mat W = params->W;
  

  //apply smear + odm to hdof
  PetscInt nhdof = nr*ncells*nlayers_active;
  PetscInt nhdof_odm = (nr+pr)*nlayers_active + (pr+nr+pr)*nlayers_active*(ncells-1);
  PetscScalar *hdof=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
  PetscScalar *hdof_odm=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
  for(PetscInt i=0;i<nhdof;i++)
    hdof[i]=dof[i]+PETSC_i*0.0;
  matmult_arrays(W, hdof,SCAL, hdof_odm,SCAL, 0);

  PetscInt nsims = params->nsims;
  PetscScalar *Ex=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Ey=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Hx=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Hy=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Pz=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar **Exgrad=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Eygrad=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Hxgrad=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Hygrad=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Pzgrad=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  for(PetscInt isim=0;isim<nsims;isim++){
    Ex[isim]=0, Ey[isim]=0, Hx[isim]=0, Hy[isim]=0, Pz[isim]=0;
    Exgrad[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Eygrad[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hxgrad[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hygrad[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Pzgrad[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    for(PetscInt i=0;i<nhdof_odm;i++)
      Exgrad[isim][i]=0, Eygrad[isim][i]=0, Hxgrad[isim][i]=0, Hygrad[isim][i]=0, Pzgrad[isim][i]=0;
  }

  //divide into cells and do the simulations in parallel; adjoints computed
  MPI_Comm subcomm = params->subcomm;
  PetscInt nsolves_per_comm = params->nsolves_per_comm;
  PetscInt colour = params->colour;
  //PetscInt **idset_specs = params->idset_specs;
  //PetscInt **idset_sims = params->idset_sims;
  PetscInt ***idset_solves = params->idset_solves;
  PetscReal *freqs = params->freqs;
  PetscInt nangles = params->nangles;
  PetscInt mzsum = params->mzsum;
  PetscInt *mz = params->mz;
  PetscReal filter_beta = params->filter_beta;
  PetscInt zfixed = params->zfixed;
  for(PetscInt isolve_per_comm=0;isolve_per_comm<nsolves_per_comm;isolve_per_comm++){

    PetscInt isolve  = isolve_per_comm + nsolves_per_comm * colour;
    PetscInt msolves = idset_solves[isolve][0][0];
    PetscInt ifreq   = idset_solves[isolve][0][1];
    PetscInt icell   = idset_solves[isolve][0][2];
    PetscInt m       = idset_solves[isolve][0][3];

    Mat A = params->A[isolve_per_comm];
    Mat DDe = params->DDe[isolve_per_comm];
    Mat Qp = params->Qp[isolve_per_comm];
    Mat Qp_minus = params->Qp_minus[isolve_per_comm];
    KSP ksp = params->ksp[isolve_per_comm];
    PetscInt *its = &(params->its[isolve_per_comm]);
    Vec epsBkg = params->epsBkg[isolve_per_comm];
    Vec epsDiff = params->epsDiff[isolve_per_comm];
    PetscReal omega = 2.0*M_PI*freqs[ifreq];
    
    PetscInt nr_cell = (icell==0) ? nr+pr : pr+nr+pr;
    PetscInt cell_start = (icell==0) ? 0 : ((nr+pr) + (pr+nr+pr)*(icell-1))*nlayers_active;
    PetscInt nedof = nr_cell*mzsum;
    PetscScalar *edof = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
    varh_expand(&(hdof_odm[cell_start]), edof, nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);
    PetscInt nz = params->nz;
    PetscInt iz_src = params->iz_src;
    //PetscInt iz_mtr = params->iz_mtr;
    PetscInt maxit = params->maxit;
        
    for(PetscInt jsolve=0;jsolve<msolves;jsolve++){

      //PetscInt ispec=idset_solves[isolve][1][jsolve];
      PetscInt isim=idset_solves[isolve][2][jsolve];

      PetscScalar *Jr = params->Jr[isolve_per_comm][jsolve];
      PetscScalar *Jt = params->Jt[isolve_per_comm][jsolve];
      
      Vec gex = params->gex[jsolve + 2*nangles * isolve_per_comm];
      Vec gey = params->gey[jsolve + 2*nangles * isolve_per_comm];
      Vec ghx = params->ghx[jsolve + 2*nangles * isolve_per_comm];
      Vec ghy = params->ghy[jsolve + 2*nangles * isolve_per_comm];
      Vec gex_minus = params->gex_minus[jsolve + 2*nangles * isolve_per_comm];
      Vec gey_minus = params->gey_minus[jsolve + 2*nangles * isolve_per_comm];
      Vec ghx_minus = params->ghx_minus[jsolve + 2*nangles * isolve_per_comm];
      Vec ghy_minus = params->ghy_minus[jsolve + 2*nangles * isolve_per_comm];
      Vec sigmahat =  params->sigmahat[jsolve + 2*nangles * isolve_per_comm];
      
      PetscScalar tmpEx, tmpEy, tmpHx, tmpHy, tmpPz;
      PetscScalar *tmpEx_egrad = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
      PetscScalar *tmpEy_egrad = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
      PetscScalar *tmpHx_egrad = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
      PetscScalar *tmpHy_egrad = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));
      PetscScalar *tmpPz_egrad = (PetscScalar *)malloc(nedof*sizeof(PetscScalar));	    
      PetscInt nhgrad = nr_cell*nlayers_active;
      PetscScalar *tmpEx_hgrad = (PetscScalar *)malloc(nhgrad*sizeof(PetscScalar));
      PetscScalar *tmpEy_hgrad = (PetscScalar *)malloc(nhgrad*sizeof(PetscScalar));
      PetscScalar *tmpHx_hgrad = (PetscScalar *)malloc(nhgrad*sizeof(PetscScalar));
      PetscScalar *tmpHy_hgrad = (PetscScalar *)malloc(nhgrad*sizeof(PetscScalar));
      PetscScalar *tmpPz_hgrad = (PetscScalar *)malloc(nhgrad*sizeof(PetscScalar));
      strehlpieces(subcomm, nr_cell,nz, edof, A,
		   DDe, epsDiff, epsBkg, omega, 
		   ksp, its, maxit, 
		   Jr, Jt, iz_src, 
		   gex, gey, ghx, ghy, Qp, 
		   &tmpEx, &tmpEy, &tmpHx, &tmpHy, &tmpPz,
		   tmpEx_egrad, tmpEy_egrad,
		   tmpHx_egrad, tmpHy_egrad,
		   tmpPz_egrad,
		   m, sigmahat,
		   gex_minus,gey_minus,ghx_minus,ghy_minus,Qp_minus);
      varh_contract(tmpEx_egrad, tmpEx_hgrad, &(hdof_odm[cell_start]), nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);
      varh_contract(tmpEy_egrad, tmpEy_hgrad, &(hdof_odm[cell_start]), nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);
      varh_contract(tmpHx_egrad, tmpHx_hgrad, &(hdof_odm[cell_start]), nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);
      varh_contract(tmpHy_egrad, tmpHy_hgrad, &(hdof_odm[cell_start]), nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);
      varh_contract(tmpPz_egrad, tmpPz_hgrad, &(hdof_odm[cell_start]), nr_cell, 1, nlayers_active, mz, filter_beta, zfixed);

      PetscInt rank;
      MPI_Comm_rank(subcomm,&rank);
      if(rank==0){
	Ex[isim]=tmpEx, Ey[isim]=tmpEy, Hx[isim]=tmpHx, Hy[isim]=tmpHy, Pz[isim]=tmpPz;
	for(PetscInt i=0;i<nhgrad;i++){
	  Exgrad[isim][cell_start+i]=tmpEx_hgrad[i];
	  Eygrad[isim][cell_start+i]=tmpEy_hgrad[i];
	  Hxgrad[isim][cell_start+i]=tmpHx_hgrad[i];
	  Hygrad[isim][cell_start+i]=tmpHy_hgrad[i];
	  Pzgrad[isim][cell_start+i]=tmpPz_hgrad[i];
	}
      }
      
      free(tmpEx_egrad);
      free(tmpEy_egrad);
      free(tmpHx_egrad);
      free(tmpHy_egrad);
      free(tmpPz_egrad);
      free(tmpEx_hgrad);
      free(tmpEy_hgrad);
      free(tmpHx_hgrad);
      free(tmpHy_hgrad);
      free(tmpPz_hgrad);

    }

    free(edof);
    
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"SUCCESS: all simulations done!\n");
  
  //gather
  PetscScalar *Ex_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Ey_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Hx_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Hy_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *Pz_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar **Exgrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Eygrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Hxgrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Hygrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **Pzgrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  for(PetscInt isim=0;isim<nsims;isim++){
    Exgrad_all[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Eygrad_all[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hxgrad_all[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hygrad_all[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Pzgrad_all[isim]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
  }
  MPI_Allreduce(Ex,Ex_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(Ey,Ey_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(Hx,Hx_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(Hy,Hy_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(Pz,Pz_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  for(PetscInt isim=0;isim<nsims;isim++){
    MPI_Allreduce(Exgrad[isim],Exgrad_all[isim],nhdof_odm,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(Eygrad[isim],Eygrad_all[isim],nhdof_odm,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(Hxgrad[isim],Hxgrad_all[isim],nhdof_odm,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(Hygrad[isim],Hygrad_all[isim],nhdof_odm,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(Pzgrad[isim],Pzgrad_all[isim],nhdof_odm,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  free(Ex);
  free(Ey);
  free(Hx);
  free(Hy);
  free(Pz);
  for(PetscInt isim=0;isim<nsims;isim++){
    free(Exgrad[isim]);
    free(Eygrad[isim]);
    free(Hxgrad[isim]);
    free(Hygrad[isim]);
    free(Pzgrad[isim]);
  }
  free(Exgrad);
  free(Eygrad);
  free(Hxgrad);
  free(Hygrad);
  free(Pzgrad);
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"SUCCESS: all simulations gathered!\n");
  
  //consolidate
  PetscInt **num_m=params->num_m;
  PetscScalar Ex_specs[nspecs];
  PetscScalar Ey_specs[nspecs];
  PetscScalar Hx_specs[nspecs];
  PetscScalar Hy_specs[nspecs];
  PetscScalar Pz_specs[nspecs];
  PetscScalar *Exgrad_specs[nspecs];
  PetscScalar *Eygrad_specs[nspecs];
  PetscScalar *Hxgrad_specs[nspecs];
  PetscScalar *Hygrad_specs[nspecs];
  PetscScalar *Pzgrad_specs[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    Ex_specs[ispec]=0;
    Ey_specs[ispec]=0;
    Hx_specs[ispec]=0;
    Hy_specs[ispec]=0;
    Pz_specs[ispec]=0;
    Exgrad_specs[ispec]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Eygrad_specs[ispec]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hxgrad_specs[ispec]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Hygrad_specs[ispec]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    Pzgrad_specs[ispec]=(PetscScalar *)malloc(nhdof_odm*sizeof(PetscScalar));
    for(PetscInt i=0;i<nhdof_odm;i++){
      Exgrad_specs[ispec][i]=0;
      Eygrad_specs[ispec][i]=0;
      Hxgrad_specs[ispec][i]=0;
      Hygrad_specs[ispec][i]=0;
      Pzgrad_specs[ispec][i]=0;
    }
  }
  PetscInt isim=0;
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    for(PetscInt icell=0;icell<ncells;icell++){
      for(PetscInt im=0;im<num_m[ispec][icell];im++){
	
	Ex_specs[ispec] += Ex_all[isim];
	Ey_specs[ispec] += Ey_all[isim];
	Hx_specs[ispec] += Hx_all[isim];
	Hy_specs[ispec] += Hy_all[isim];
	Pz_specs[ispec] += Pz_all[isim];
	PetscInt nr_cell = (icell==0) ? nr+pr : pr+nr+pr;
	PetscInt block_size = nr_cell * nlayers_active;
	PetscInt cell_start = (icell==0) ? 0 : ((nr+pr) + (pr+nr+pr)*(icell-1))*nlayers_active;
	for(PetscInt i=0;i<block_size;i++){
	  Exgrad_specs[ispec][cell_start+i] += Exgrad_all[isim][cell_start+i];
	  Eygrad_specs[ispec][cell_start+i] += Eygrad_all[isim][cell_start+i];
	  Hxgrad_specs[ispec][cell_start+i] += Hxgrad_all[isim][cell_start+i];
	  Hygrad_specs[ispec][cell_start+i] += Hygrad_all[isim][cell_start+i];
	  Pzgrad_specs[ispec][cell_start+i] += Pzgrad_all[isim][cell_start+i];
	}
	
	isim++;
      }
    }
  }
  free(Ex_all);
  free(Ey_all);
  free(Hx_all);
  free(Hy_all);
  free(Pz_all);
  for(PetscInt isim=0;isim<nsims;isim++){
    free(Exgrad_all[isim]);
    free(Eygrad_all[isim]);
    free(Hxgrad_all[isim]);
    free(Hygrad_all[isim]);
    free(Pzgrad_all[isim]);
  }
  free(Exgrad_all);
  free(Eygrad_all);
  free(Hxgrad_all);
  free(Hygrad_all);
  free(Pzgrad_all);
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"SUCCESS: all simulations consolidated!\n");
  
  //back prop thru odm+smear 
  PetscScalar *dEx_specs[nspecs];
  PetscScalar *dEy_specs[nspecs];
  PetscScalar *dHx_specs[nspecs];
  PetscScalar *dHy_specs[nspecs];
  PetscScalar *dPz_specs[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    dEx_specs[ispec]=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
    dEy_specs[ispec]=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
    dHx_specs[ispec]=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
    dHy_specs[ispec]=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
    dPz_specs[ispec]=(PetscScalar *)malloc(nhdof*sizeof(PetscScalar));
    matmult_arrays(W,Exgrad_specs[ispec],SCAL,dEx_specs[ispec],SCAL, 1);
    matmult_arrays(W,Eygrad_specs[ispec],SCAL,dEy_specs[ispec],SCAL, 1);
    matmult_arrays(W,Hxgrad_specs[ispec],SCAL,dHx_specs[ispec],SCAL, 1);
    matmult_arrays(W,Hygrad_specs[ispec],SCAL,dHy_specs[ispec],SCAL, 1);
    matmult_arrays(W,Pzgrad_specs[ispec],SCAL,dPz_specs[ispec],SCAL, 1);
    free(Exgrad_specs[ispec]);
    free(Eygrad_specs[ispec]);
    free(Hxgrad_specs[ispec]);
    free(Hygrad_specs[ispec]);
    free(Pzgrad_specs[ispec]);
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"SUCCESS: back prop done!\n");

  //populate the real-valued result and grad;
  PetscReal epi_t = dof[ndof-1];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    PetscReal airyfactor = params->airyfactor[ispec];
    PetscReal Sz = creal( Ex_specs[ispec] * conj(Hy_specs[ispec]) - Ey_specs[ispec] * conj(Hx_specs[ispec]) );
    PetscReal Pz = creal( Pz_specs[ispec] );
    result[ispec]=epi_t - Sz/Pz * airyfactor;
    for(PetscInt idof=0;idof<nhdof;idof++){
      PetscReal dSz = creal( dEx_specs[ispec][idof] * conj(Hy_specs[ispec]) + Ex_specs[ispec] * conj(dHy_specs[ispec][idof]) - \
			     dEy_specs[ispec][idof] * conj(Hx_specs[ispec]) - Ey_specs[ispec] * conj(dHx_specs[ispec][idof]) );
      PetscReal dPz = creal( dPz_specs[ispec][idof] );
      grad[idof + ndof*ispec] = - ( dSz/Pz - (Sz/pow(Pz,2)) * dPz ) * airyfactor;
    }
    grad[nhdof + ndof*ispec]=1.0;
    free(dEx_specs[ispec]);
    free(dEy_specs[ispec]);
    free(dHx_specs[ispec]);
    free(dHy_specs[ispec]);
    free(dPz_specs[ispec]);
  }
  MPI_Barrier(PETSC_COMM_WORLD);

  for(PetscInt ispec=0;ispec<nspecs;ispec++)
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: the objective value for spec %d at step %d is %0.16g \n", ispec, count, epi_t - result[ispec]);

  free(hdof);
  free(hdof_odm);
  MPI_Barrier(PETSC_COMM_WORLD);


}

