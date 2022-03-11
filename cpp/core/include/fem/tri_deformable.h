#ifndef FEM_TRI_DEFORMABLE_H
#define FEM_TRI_DEFORMABLE_H

#include "fem/deformable.h"

class TriDeformable : public Deformable<2, 3> {
protected:
    void InitializeFiniteElementSamples() override;
    const int GetNumOfSamplesInElement() const { return 1; }
};

#endif