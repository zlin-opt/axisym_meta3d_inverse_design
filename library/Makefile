export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o

NLOPT_INC = -I /usr/local/include

CC=mpicc
CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${NLOPT_INC} ${PETSC_CC_INCLUDES}

LIBS4OBJS=maxwell.o input.o output.o solver.o array2vec.o dof2dom.o ovmat.o varh.o pwsrc.o cubature.o vec.o misc.o optitemp.o greendyadic.o qforms.o adjoint.o strehl.o 

lib: libCYLOPT.a $(LIBS4OBJS)

libCYLOPT.a: $(LIBS4OBJS)
	$(AR) crvs $@ *.o


