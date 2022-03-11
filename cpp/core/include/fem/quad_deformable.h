#ifndef FEM_QUAD_DEFORMABLE_H
#define FEM_QUAD_DEFORMABLE_H

#include "fem/deformable.h"

class QuadDeformable : public Deformable<2, 4> {
protected:
    void InitializeFiniteElementSamples() override;
    const int GetNumOfSamplesInElement() const { return 4; }
};

#endif