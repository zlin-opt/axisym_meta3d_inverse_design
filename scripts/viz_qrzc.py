import numpy as np
import h5py as hp
import sys

prefix=sys.argv[1]
Nr=int(sys.argv[2])
Nz=int(sys.argv[3])

filename=prefix+'.qrzc'
F=np.loadtxt(filename)

Fr=np.zeros((Nr,Nz),dtype=np.complex64)
Ft=np.zeros((Nr,Nz),dtype=np.complex64)
Fz=np.zeros((Nr,Nz),dtype=np.complex64)

for ic in range(3):
    for iz in range(Nz):
        for ir in range(Nr):
            i=ir + Nr*iz + Nr*Nz*ic
            if ic==0:
                Fr[ir,iz]=F[2*i+0] + 1j*F[2*i+1]
            if ic==1:
                Ft[ir,iz]=F[2*i+0] + 1j*F[2*i+1]
            if ic==2:
                Fz[ir,iz]=F[2*i+0] + 1j*F[2*i+1]

fid=hp.File(prefix+'.h5','w')
fid.create_dataset('Fr.real',data=np.real(Fr))
fid.create_dataset('Fr.imag',data=np.imag(Fr))
fid.create_dataset('Ft.real',data=np.real(Ft))
fid.create_dataset('Ft.imag',data=np.imag(Ft))
fid.create_dataset('Fz.real',data=np.real(Fz))
fid.create_dataset('Fz.imag',data=np.imag(Fz))
fid.close()
