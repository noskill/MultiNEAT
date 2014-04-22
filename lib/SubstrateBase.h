#ifndef BASESUBSTRATE_H
#define BASESUBSTRATE_H

namespace NEAT{


class SubstrateBase
{
public:

    // Return the minimum input dimensionality of the CPPN
    virtual unsigned int GetMinCPPNInputs();

    // Return the minimum output dimensionality of the CPPN
    virtual unsigned int GetMinCPPNOutputs();
protected:

    virtual unsigned int GetMaxDims()=0;
};

}

#endif // BASESUBSTRATE_H
