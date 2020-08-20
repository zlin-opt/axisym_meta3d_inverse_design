import numpy as np
import h5py as hp
import sys
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import scipy

filename=sys.argv[1]
outputfilename=sys.argv[2]
freq=float(sys.argv[3])
Pznear=float(sys.argv[4])
Nx=500
Nxfar=1000
dx=0.04
dxfar=0.04
flen=250
NA=np.sin(np.arctan(Nx*dx/flen))
FWHM=1.0/(2*freq*NA)

xfar=np.linspace(-Nxfar*dxfar/2,Nxfar*dxfar/2,Nxfar)

Sy = np.loadtxt(filename)
Symax = np.max(Sy)
Sy=Sy/Symax
tmpSy=Sy-0.5

Syinterp=interp1d(xfar,tmpSy)
x0=fsolve(Syinterp,-FWHM/2)[0]
x1=fsolve(Syinterp, FWHM/2)[0]

print "\n"

print "Designed NA: "+str(NA)
print "Designed FWHM: "+str(FWHM)
print "Measured FWHM: "+str(x1-x0)

fwhm=x1-x0

print "\n"

np.savetxt(outputfilename+'.dat',Sy)

na = 1.0/(2*freq*fwhm)
idealairy=np.zeros(Nxfar)
i=0
for xx in xfar:
    r=6.46536*na*xx*freq
    idealairy[i]=np.power(2*scipy.special.jv(1,r)/r,2)
    i=i+1
    
np.savetxt('idealairy.dat',idealairy)

strehl=4*np.pi/np.power(6.46536*freq*na,2) * Symax/Pznear
print "Strehl ratio with measured FWHM: " + str(strehl)
