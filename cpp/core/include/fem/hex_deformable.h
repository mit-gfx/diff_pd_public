#ifndef FEM_HEX_DEFORMABLE_H
#define FEM_HEX_DEFORMABLE_H

#include "fem/deformable.h"

class HexDeformable : public Deformable<3, 8> {
protected:
    void InitializeFiniteElementSamples() override;
    const int GetNumOfSamplesInElement() const { return 8; }
};

#endif