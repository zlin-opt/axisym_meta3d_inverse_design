export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o *_exec

CC=mpicc

CYLOPT_INC = -I ../library/
CYLOPT_LIB = ../library/libCYLOPT.a

NLOPT_INC = -I /usr/local/include
NLOPT_LIB = /usr/local/lib/libnlopt.a

CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${CYLOPT_INC} ${NLOPT_INC} ${PETSC_CC_INCLUDES} 
LIBS=$(CYLOPT_LIB) $(NLOPT_LIB) $(PETSC_LIB)

all: strehlopt_exec

strehlopt.o: strehlopt.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
strehlopt_exec: strehlopt.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
