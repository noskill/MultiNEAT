#include "SubstrateBase.h"

using namespace NEAT;

unsigned int SubstrateBase::GetMinCPPNInputs()
{
    // determine the dimensionality across the entire substrate
    unsigned int cppn_inputs = GetMaxDims() * 2; // twice, because we query 2 points at a time
    return cppn_inputs + 1; // always count the bias
}


unsigned int SubstrateBase::GetMinCPPNOutputs()
{
   return 1;
}
