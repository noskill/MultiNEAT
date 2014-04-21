#ifndef BASESUBSTRATE_H
#define BASESUBSTRATE_H

namespace NEAT{


class BaseSubstrate
{
    // Return the minimum input dimensionality of the CPPN
    virtual unsigned int GetMinCPPNInputs()=0;

    // Return the minimum output dimensionality of the CPPN
    virtual unsigned int GetMinCPPNOutputs()=0;

    virtual unsigned int GetMinCPPNInputs();

    virtual unsigned int GetMaxDims();
};

}

#endif // BASESUBSTRATE_H
