#ifndef FEM_TET_DEFORMABLE_H
#define FEM_TET_DEFORMABLE_H

#include "fem/deformable.h"

class TetDeformable : public Deformable<3, 4> {
protected:
    void InitializeFiniteElementSamples() override;
    const int GetNumOfSamplesInElement() const { return 1; }
};

#endif