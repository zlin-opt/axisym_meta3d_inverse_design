#ifndef GUARD_strehl_h
#define GUARD_strehl_h

#include <assert.h>
#include "petsc.h"
#include "array2vec.h"
#include "vec.h"
#include "solver.h"
#include "obj.h"
#include "varh.h"
#include "misc.h"
#include "type.h"
#include "adjoint.h"
#include "output.h"

void strehl(unsigned nspecs, PetscReal *result,
	    unsigned ndof, const PetscReal *dof,
	    PetscReal *grad,
	    void *data);

#endif
