import sys
import numpy as np
#import h5py as hp
import os
#from scipy.interpolate import interp1d
from random import random

################################
def writeflag(fid,pre,x):
    if hasattr(x, "__len__" ):
        strx=pre+" "+str(x[0])
        for y in x[1::]:
            strx+=","+str(y)
    else:
        strx=pre+" "+str(x)
    strx+="\n"
    fid.write(strx)
    return strx

def mat_sio2(lamnm):
    lam=lamnm/1000.0
    epsilon=1 + 0.6961663*lam**2/(lam**2-0.0684043**2) + 0.4079426*lam**2/(lam**2-0.1162414**2) + 0.8974794*lam**2/(lam**2-9.896161**2)
    return epsilon

# def mat_tio2(lamnm):
    
#     file='tio2_refractiveindex_[nm].txt'
#     data=np.loadtxt(file)

#     n=interp1d(data[:,0],data[:,1])(lamnm)
#     eps=n**2
#     return eps

def mat_ipdip(lamum):

    return ( 1.5273 + 6.5456*10**(-3)/(lamum**2) + 2.5345*10**(-4)/(lamum**4) )**2

class grid:
    def __init__(self, bc,xraw,yraw,zraw,mpml,ekl):
        self.bc=bc
        self.xraw=xraw
        self.yraw=yraw
        self.zraw=zraw
        self.mpml=mpml
        self.ekl=ekl

    def printgrid(self,name):
        fid=hp.File(name,"w")
        fid.create_dataset("bc",data=self.bc)
        fid.create_dataset("xraw",data=self.xraw)
        fid.create_dataset("yraw",data=self.yraw)
        fid.create_dataset("zraw",data=self.zraw)
        fid.create_dataset("Mpml",data=self.mpml)
        fid.create_dataset("e_ikL",data=self.ekl)
        fid.close()
                                                                                                                                               

###############################

nlayers=5

fid=open("config","w")

ncells=1
ncolors=10
wavmax=1600
wavmin=1300
colors=np.round(np.linspace(wavmax,wavmin,ncolors),3)
nangles=10
angmax=5
angmin=0
angles=np.round(np.linspace(angmin,angmax,nangles),3)
npols=2*np.ones(nangles,dtype=int)
npols[0]=1

nr=2500
pr=60
pmlr=50
dr=0.01
dz=0.01

foclen=200
lens_radius=nr*ncells*dr
NA=np.sin(np.arctan(lens_radius/foclen))
lensid="### fmax=%g (%g nm), fmin=%g (%g nm), focallength=%g, radius=%g, NA=%g, %d colors, %d angles\n" % (wavmin/wavmin, wavmin, wavmin/wavmax, wavmax, foclen, lens_radius, NA, ncolors, nangles)
fid.write(lensid)
writeflag(fid,"# ncells*ncolors*nangles = ",ncells*ncolors*nangles)
writeflag(fid,"-nr",nr)
writeflag(fid,"-ncells",ncells)
writeflag(fid,"-dr",dr)
writeflag(fid,"-pr",pr)
writeflag(fid,"-pmlr",pmlr)
writeflag(fid,"-dz",dz)
writeflag(fid,"-focal_length",foclen)

#alternating patterened and uniform layers; same thickness
#nlayers=8
nlayers_tot=2*nlayers+1
pmlz=50
pml2src=6
src2stk=36
stk2ref=36
ref2pml=6
active_layer_thickness=60
mid_layer_thickness=1

tsub=pmlz+pml2src+src2stk
tsup=stk2ref+ref2pml+pmlz
thickness=np.zeros(nlayers_tot,dtype=int)
for i in range(nlayers_tot):
    if i%2==0:
        thickness[i]=mid_layer_thickness
    if i%2==1:
        thickness[i]=active_layer_thickness
    if i==0:
        thickness[i]=tsub
    if i==nlayers_tot - 1:
        thickness[i]=tsup
id_active_layers=[ 2*i+1 for i in range(nlayers) ]

epssio2=[round(mat_sio2(color),3) for color in colors]
freqs=[ round(colors[ncolors-1]/colors[i],3) for i in range(ncolors) ]

writeflag(fid,"-pmlz",pmlz)
writeflag(fid,"-nlayers_total",nlayers_tot)
writeflag(fid,"-nlayers_active",nlayers)
writeflag(fid,"-thickness",thickness)
writeflag(fid,"-id_active_layers",id_active_layers)

writeflag(fid,"-nfreqs",ncolors)
writeflag(fid,"-freqs",freqs)
writeflag(fid,"-nangles",nangles)
writeflag(fid,"-angles",angles)
writeflag(fid,"-npols",npols)

epsbkg=np.zeros((ncolors,nlayers_tot*2))
epsfeg=np.zeros((ncolors,nlayers*2))
for ifreq in range(ncolors):
    for i in range(nlayers_tot):
        if i%2==0:
            epsbkg[ifreq,i]=epssio2[ifreq]
        if i%2==1:
            epsbkg[ifreq,i]=1
        if i==0:
            epsbkg[ifreq,i]=epssio2[ifreq]
        if i==nlayers_tot - 1:
            epsbkg[ifreq,i]=1
    writeflag(fid,"-ifreq"+str(ifreq)+"_epsbkg",epsbkg[ifreq,:])
    epsfeg[ifreq,0:nlayers]=epssio2[ifreq]*np.ones(nlayers)
    epsfeg[ifreq,nlayers:2*nlayers]=0
    writeflag(fid,"-ifreq"+str(ifreq)+"_epsfeg",epsfeg[ifreq,:])

writeconfigs=0
if writeconfigs==1:
    for i2 in range(nangles):
        for i1 in range(ncolors):
            i=i1+ncolors*i2
            ffid=open('config'+str(i),'w')
            writeflag(ffid,"-nr",nr)
            writeflag(ffid,"-ncells",ncells)
            writeflag(ffid,"-dr",dr)
            writeflag(ffid,"-pr",pr)
            writeflag(ffid,"-pmlr",pmlr)
            writeflag(ffid,"-dz",dz)
            writeflag(ffid,"-focal_length",foclen)
            writeflag(ffid,"-pmlz",pmlz)
            writeflag(ffid,"-nlayers_total",nlayers_tot)
            writeflag(ffid,"-nlayers_active",nlayers)
            writeflag(ffid,"-thickness",thickness)
            writeflag(ffid,"-id_active_layers",id_active_layers)
            writeflag(ffid,"-nfreqs",1)
            writeflag(ffid,"-freqs",freqs[i1])
            writeflag(ffid,"-nangles",1)
            writeflag(ffid,"-angles",angles[i2])
            writeflag(ffid,"-ifreq0_epsbkg",epsbkg[i1,:])
            writeflag(ffid,"-ifreq0_epsfeg",epsfeg[i1,:])
        
initfile1="mid.txt"
initfile2="blank.txt"
initfile3="full.txt"
initfile4="rand.txt"
dof1=np.zeros(nr*ncells*nlayers)
dof2=np.zeros(nr*ncells*nlayers)
dof3=np.zeros(nr*ncells*nlayers)
dof4=np.zeros(nr*ncells*nlayers)
for il in range(nlayers):
    for ic in range(ncells):
        for ix in range(nr):
            i=ix + nr*ic + nr*ncells*il
            dof1[i]=0.5
            dof2[i]=0.01
            dof3[i]=1.0
            dof4[i]=random()
np.savetxt(initfile1,dof1)
np.savetxt(initfile2,dof2)
np.savetxt(initfile3,dof3)
np.savetxt(initfile4,dof4)

