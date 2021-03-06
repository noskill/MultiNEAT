///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Phenotype.cpp
// Description: Implementation of the phenotype activation functions.
///////////////////////////////////////////////////////////////////////////////



#include <math.h>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "NeuralNetwork.h"
#include "Assert.h"
#include "Utils.h"
#include <boost/format.hpp>

#define sqr(x) ((x)*(x))
#define LEARNING_RATE 0.0001

namespace NEAT
{

/////////////////////////////////////
// The set of activation functions //
/////////////////////////////////////


inline double af_sigmoid_unsigned(double aX, double aSlope, double aShift)
{
    return 1.0 / (1.0 + exp( - aSlope * aX - aShift));
}

inline double af_sigmoid_signed(double aX, double aSlope, double aShift)
{
    double tY = af_sigmoid_unsigned(aX, aSlope, aShift);
    return (tY - 0.5) * 2.0;
}


inline double af_tanh(double aX, double aSlope, double aShift)
{
    return tanh(aX * aSlope);
}

inline double af_tanh_cubic(double aX, double aSlope, double aShift)
{
    return tanh(aX * aX * aX * aSlope);
}

inline double af_step_signed(double aX, double aShift)
{
    double tY;
    if (aX > aShift)
    {
        tY = 1.0;
    }
    else
    {
        tY = -1.0;
    }

    return tY;
}

inline double af_step_unsigned(double aX, double aShift)
{
    if (aX > aShift)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

inline double af_gauss_signed(double aX, double aSlope, double aShift)
{
    double tY = exp( - aSlope * aX * aX + aShift);
    return (tY-0.5)*2.0;
}

inline double af_gauss_unsigned(double aX, double aSlope, double aShift)
{
    return exp( - aSlope * aX * aX + aShift);
}

inline double af_abs(double aX, double aShift)
{
    return ((aX + aShift)< 0.0)? -(aX + aShift): (aX + aShift);
}

inline double af_sine_signed(double aX, double aFreq, double aShift)
{
    aFreq = 3.141592;
    return sin(aX * aFreq + aShift);
}

inline double af_sine_unsigned(double aX, double aFreq, double aShift)
{
    double tY = sin((aX * aFreq + aShift) );
    return (tY + 1.0) / 2.0;
}

inline double af_square_signed(double aX, double aHighPulseSize, double aLowPulseSize)
{
    return 0.0;    // TODO
}

inline double af_square_unsigned(double aX, double aHighPulseSize, double aLowPulseSize)
{
    return 0.0;    // TODO
}

inline double af_linear(double aX, double aShift)
{
    return (aX + aShift);
}



double unsigned_sigmoid_derivative(double x)
{
    return x * (1 - x);
}

double tanh_derivative(double x)
{
    return 1 - x * x;
}

///////////////////////////////////////
// Neural network class implementation
///////////////////////////////////////
NeuralNetwork::NeuralNetwork(bool a_Minimal):
    m_Depth(0), is_depth_ready(false)
{
    if (!a_Minimal)
    {
        // build an XOR network

        // The input neurons are 3 // indexes 0 1 2
        Neuron t_i1, t_i2, t_i3;

        // The output neuron       // index 3
        Neuron t_o1;

        // The hidden neuron       // index 4
        Neuron t_h1;

        m_neurons[0] = t_i1;
        m_neurons[1] = (t_i2);
        m_neurons[2] = (t_i3);
        m_neurons[3] = (t_o1);
        m_neurons[4] = (t_h1);

        // The connections
        Connection t_c;

        t_c.m_source_neuron_idx = 0;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 1;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 2;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 0;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 1;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 2;
        t_c.m_target_neuron_idx = 4;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        t_c.m_source_neuron_idx = 4;
        t_c.m_target_neuron_idx = 3;
        t_c.m_weight = 0;
        m_connections.push_back(t_c);

        m_num_inputs = 3;
        m_num_outputs = 1;

        // Initialize the network's weights (make them random)
        for (unsigned int i = 0; i < m_connections.size(); i++)
        {
            m_connections[i].m_weight = ((double) rand() / (double) RAND_MAX)
                    - 0.5;
        }

        // clean up other neuron data as well
        for (unsigned int i = 0; i < m_neurons.size(); i++)
        {
            m_neurons[i].m_a = 1;
            m_neurons[i].m_b = 0;
            m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                    m_neurons[i].m_membrane_potential = 0;
        }

        InitRTRLMatrix();
    }
    else
    {
        // an empty network
        m_num_inputs = m_num_outputs = 0;
        m_total_error = 0;
        // clean up other neuron data as well
        for (unsigned int i = 0; i < m_neurons.size(); i++)
        {
            m_neurons[i].m_a = 1;
            m_neurons[i].m_b = 0;
            m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                    m_neurons[i].m_membrane_potential = 0;
        }
        Clear();
    }
}

NeuralNetwork::NeuralNetwork():
    m_Depth(0), is_depth_ready(false)
{
    // an empty network
    m_num_inputs = m_num_outputs = 0;
    m_total_error = 0;
    // clean up other neuron data as well
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_a = 1;
        m_neurons[i].m_b = 0;
        m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                m_neurons[i].m_membrane_potential = 0;
    }
    Clear();
}

void NeuralNetwork::InitRTRLMatrix()
{
    // Allocate memory for the neurons sensitivity matrices.
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_sensitivity_matrix.resize(m_neurons.size()); // first dimention
        for (unsigned int j = 0; j < m_neurons.size(); j++)
        {
            m_neurons[i].m_sensitivity_matrix[j].resize(m_neurons.size()); // second dimention
        }
    }

    // now clear it
    FlushCube();
    // clear out the other RTRL stuff as well
    m_total_error = 0;
    m_total_weight_change.resize(m_connections.size());
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_total_weight_change[i] = 0;
    }
}

void NeuralNetwork::ActivateFast()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_activesum;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        m_neurons[i].m_activation = y;
    }
}

void NeuralNetwork::Activate()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
        CalculateNeuronActivation(i);

}

void NeuralNetwork::CalculateNeuronActivation(size_t i){
    double x = m_neurons[i].m_activesum;
    m_neurons[i].m_activesum = 0;
    // Apply the activation function
    double y = 0.0;
    switch (m_neurons[i].m_activation_function_type)
    {
    case SIGNED_SIGMOID:
        y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case UNSIGNED_SIGMOID:
        y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case TANH:
        y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case TANH_CUBIC:
        y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case SIGNED_STEP:
        y = af_step_signed(x, m_neurons[i].m_b);
        break;
    case UNSIGNED_STEP:
        y = af_step_unsigned(x, m_neurons[i].m_b);
        break;
    case SIGNED_GAUSS:
        y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case UNSIGNED_GAUSS:
        y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case ABS:
        y = af_abs(x, m_neurons[i].m_b);
        break;
    case SIGNED_SINE:
        y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case UNSIGNED_SINE:
        y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case SIGNED_SQUARE:
        y = af_square_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case UNSIGNED_SQUARE:
        y = af_square_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    case LINEAR:
        y = af_linear(x, m_neurons[i].m_b);
        break;
    default:
        y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
        break;
    }
    m_neurons[i].m_activation = y;
}

void NeuralNetwork::ActivateUseInternalBias()
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_activesum + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SIGMOID:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH:
            y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH_CUBIC:
            y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_STEP:
            y = af_step_signed(x, m_neurons[i].m_b);
            break;
        case UNSIGNED_STEP:
            y = af_step_unsigned(x, m_neurons[i].m_b);
            break;
        case SIGNED_GAUSS:
            y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_GAUSS:
            y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case ABS:
            y = af_abs(x, m_neurons[i].m_b);
            break;
        case SIGNED_SINE:
            y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SINE:
            y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_SQUARE:
            y = af_square_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SQUARE:
            y = af_square_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case LINEAR:
            y = af_linear(x, m_neurons[i].m_b);
            break;
        default:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        }
        m_neurons[i].m_activation = y;
    }

}


void NeuralNetwork::ActivateLeaky(double a_dtime)
{
    // Loop connections. Calculate each connection's output signal.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_signal =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                        * m_connections[i].m_weight;
    }
    // Loop the connections again. This time add the signals to the target neurons.
    // This will largely require out of order memory writes. This is the one loop where
    // this will happen.
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                m_connections[i].m_signal;
    }
    // Now we have the leaky integrator step for the neurons
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double t_const = a_dtime / m_neurons[i].m_timeconst;
        m_neurons[i].m_membrane_potential = (1.0 - t_const)
                * m_neurons[i].m_membrane_potential
                + t_const * m_neurons[i].m_activesum;
    }
    // Now loop nodes_activesums, pass the signals through the activation function
    // and store the result back to nodes_activations
    // also skip inputs since they do not get an activation
    for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
    {
        double x = m_neurons[i].m_membrane_potential + m_neurons[i].m_bias;
        m_neurons[i].m_activesum = 0;
        // Apply the activation function
        double y = 0.0;
        switch (m_neurons[i].m_activation_function_type)
        {
        case SIGNED_SIGMOID:
            y = af_sigmoid_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SIGMOID:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH:
            y = af_tanh(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case TANH_CUBIC:
            y = af_tanh_cubic(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_STEP:
            y = af_step_signed(x, m_neurons[i].m_b);
            break;
        case UNSIGNED_STEP:
            y = af_step_unsigned(x, m_neurons[i].m_b);
            break;
        case SIGNED_GAUSS:
            y = af_gauss_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_GAUSS:
            y = af_gauss_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case ABS:
            y = af_abs(x, m_neurons[i].m_b);
            break;
        case SIGNED_SINE:
            y = af_sine_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SINE:
            y = af_sine_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case SIGNED_SQUARE:
            y = af_square_signed(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case UNSIGNED_SQUARE:
            y = af_square_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        case LINEAR:
            y = af_linear(x, m_neurons[i].m_b);
            break;
        default:
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            break;
        }
        m_neurons[i].m_activation = y;
    }

}

void NeuralNetwork::Flush()
{
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        m_neurons[i].m_activation = 0;
        m_neurons[i].m_activesum = 0;
        m_neurons[i].m_membrane_potential = 0;
    }
}

void NeuralNetwork::FlushCube()
{
    // clear the cube
    for (unsigned int i = 0; i < m_neurons.size(); i++)
        for (unsigned int j = 0; j < m_neurons.size(); j++)
            for (unsigned int k = 0; k < m_neurons.size(); k++)
                m_neurons[k].m_sensitivity_matrix[i][j] = 0;
}

void NeuralNetwork::throwWrongSize(size_t actual, size_t expected){
    throw std::runtime_error(boost::str(boost::format("input size %1% is not equal expected %2%") % actual % expected));
}

void NeuralNetwork::Input(std::vector<double>& a_Inputs)
{
    if (a_Inputs.size() != m_num_inputs)
        throwWrongSize(a_Inputs.size(), m_num_inputs);

    for (unsigned int i = 0; i < a_Inputs.size(); i++)
    {
        m_neurons[i].m_activation = a_Inputs[i];
    }
}

void NeuralNetwork::Input_python_list(py::list& a_Inputs)
{
    size_t len = py::len(a_Inputs);
    std::vector<double> inp;
    inp.resize(len);
    for(uint i=0; i<len; i++)
        inp[i] = py::extract<double>(a_Inputs[i]);

    // if the number of passed inputs differs from the actual number of inputs,
    // clip them to fit.
    if (inp.size() > m_num_inputs)
        inp.resize(m_num_inputs);
    if (inp.size() < m_num_inputs)
        throwWrongSize(len, m_num_inputs);
    Input(inp);
}

void NeuralNetwork::Input_numpy(py::numeric::array& a_Inputs)
{
    int len = py::len(a_Inputs);
    std::vector<double> inp;
    inp.resize(len);
    for(int i=0; i<len; i++)
        inp[i] = py::extract<double>(a_Inputs[i]);

    // if the number of passed inputs differs from the actual number of inputs,
    // clip them to fit.
    if (inp.size() != m_num_inputs)
        inp.resize(m_num_inputs);

    Input(inp);
}

std::vector<double> NeuralNetwork::Output()
{
    std::vector<double> t_output;
    for (int i = 0; i < m_num_outputs; i++)
    {
        t_output.push_back(m_neurons[i + m_num_inputs].m_activation);
    }
    return t_output;
}

void NeuralNetwork::Adapt(Parameters& a_Parameters)
{
    // find max absolute magnitude of the weight
    double t_max_weight = -999999999;
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        if (std::fabs(m_connections[i].m_weight) > t_max_weight)
        {
            t_max_weight = std::fabs(m_connections[i].m_weight);
        }
    }

    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        /////////////////////////////////////
        // modify weight of that connection
        ////
        double t_incoming_neuron_activation =
                m_neurons[m_connections[i].m_source_neuron_idx].m_activation;
        double t_outgoing_neuron_activation =
                m_neurons[m_connections[i].m_target_neuron_idx].m_activation;
        if (m_connections[i].m_weight > 0) // positive weight
        {
            double t_delta = (m_connections[i].m_hebb_rate
                    * (t_max_weight - m_connections[i].m_weight)
                    * t_incoming_neuron_activation
                    * t_outgoing_neuron_activation)
                    + m_connections[i].m_hebb_pre_rate * t_max_weight
                            * t_incoming_neuron_activation
                            * (t_outgoing_neuron_activation - 1.0);
            m_connections[i].m_weight = (m_connections[i].m_weight + t_delta);
        }
        else if (m_connections[i].m_weight < 0) // negative weight
        {
            // In the inhibatory case, we strengthen the synapse when output is low and
            // input is high
            double t_delta = m_connections[i].m_hebb_pre_rate
                    * (t_max_weight - m_connections[i].m_weight)
                    * t_incoming_neuron_activation
                    * (1.0 - t_outgoing_neuron_activation)
                    - m_connections[i].m_hebb_rate * t_max_weight
                            * t_incoming_neuron_activation
                            * t_outgoing_neuron_activation;
            m_connections[i].m_weight = -(m_connections[i].m_weight + t_delta);
        }

        Clamp(m_connections[i].m_weight, -a_Parameters.MaxWeight,
                a_Parameters.MaxWeight);
    }

}

int NeuralNetwork::ConnectionExists(uint a_to, uint a_from)
{
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        if ((m_connections[i].m_source_neuron_idx == a_from)
                && (m_connections[i].m_target_neuron_idx == a_to))
        {
            return i;
        }
    }

    return -1;
}

void NeuralNetwork::RTRL_update_gradients()
{
    // for every neuron
    for (unsigned int k = m_num_inputs; k < m_neurons.size(); k++)
    {
        // for all possible connections
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
            // to
            for (unsigned int j = 0; j < m_neurons.size(); j++) // from
            {
                int t_idx = ConnectionExists(i, j);
                if (t_idx != -1)
                {
                    //double t_derivative = unsigned_sigmoid_derivative( m_neurons[k].m_activation );
                    double t_derivative = 0;
                    if (m_neurons[k].m_activation_function_type
                            == NEAT::UNSIGNED_SIGMOID)
                    {
                        t_derivative = unsigned_sigmoid_derivative(
                                m_neurons[k].m_activation);
                    }
                    else if (m_neurons[k].m_activation_function_type
                            == NEAT::TANH)
                    {
                        t_derivative = tanh_derivative(
                                m_neurons[k].m_activation);
                    }

                    double t_sum = 0;
                    // calculate the other sum
                    for (unsigned int l = 0; l < m_neurons.size(); l++)
                    {
                        int t_l_idx = ConnectionExists(k, l);
                        if (t_l_idx != -1)
                        {
                            t_sum += m_connections[t_l_idx].m_weight
                                    * m_neurons[l].m_sensitivity_matrix[i][j];
                        }
                    }

                    if (i == k)
                    {
                        t_sum += m_neurons[j].m_activation;
                    }
                    m_neurons[k].m_sensitivity_matrix[i][j] = t_derivative
                            * t_sum;
                }
                else
                {
                    m_neurons[k].m_sensitivity_matrix[i][j] = 0;
                }
            }

    }

}

// please pay attention. notice here only one output is assumed
void NeuralNetwork::RTRL_update_error(double a_target)
{
    // add to total error
    m_total_error = (a_target - Output()[0]);
    // adjust each weight
    for (unsigned int i = 0; i < m_neurons.size(); i++) // to
    {
        for (unsigned int j = 0; j < m_neurons.size(); j++) // from
        {
            int t_idx = ConnectionExists(i, j);
            if (t_idx != -1)
            {
                // we know the first output's index is m_num_inputs
                double t_delta = m_total_error
                        * m_neurons[m_num_inputs].m_sensitivity_matrix[i][j];
                m_total_weight_change[t_idx] += t_delta * LEARNING_RATE;
            }
        }
    }
}

void NeuralNetwork::RTRL_update_weights()
{
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        m_connections[i].m_weight += m_total_weight_change[i];
        m_total_weight_change[i] = 0; // clear this out
    }
    m_total_error = 0;
}

void NeuralNetwork::Save(const char* a_filename)
{
    FILE* fil = fopen(a_filename, "w");
    Save(fil);
    fclose(fil);
}

void NeuralNetwork::Save(FILE* a_file)
{
    fprintf(a_file, "NNstart\n");
    // save num inputs/outputs and stuff
    fprintf(a_file, "%d %d\n", m_num_inputs, m_num_outputs);
    // save neurons
    for (unsigned int i = 0; i < m_neurons.size(); i++)
    {
        // TYPE .. A .. B .. time_const .. bias .. activation_function_type .. split_y
        fprintf(a_file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n",
                static_cast<int>(m_neurons[i].m_type), m_neurons[i].m_a,
                m_neurons[i].m_b, m_neurons[i].m_timeconst, m_neurons[i].m_bias,
                static_cast<int>(m_neurons[i].m_activation_function_type),
                m_neurons[i].m_split_y);
    }
    // save connections
    for (unsigned int i = 0; i < m_connections.size(); i++)
    {
        // from .. to .. weight.. isrecur
        fprintf(a_file, "connection %d %d %3.18f %d %3.18f %3.18f\n",
                m_connections[i].m_source_neuron_idx,
                m_connections[i].m_target_neuron_idx, m_connections[i].m_weight,
                static_cast<int>(m_connections[i].m_recur_flag),
                m_connections[i].m_hebb_rate, m_connections[i].m_hebb_pre_rate);
    }
    // end
    fprintf(a_file, "NNend\n\n");
}

bool NeuralNetwork::Load(std::ifstream& a_DataFile)
{
    std::string t_str;
    bool t_no_start = true, t_no_end = true;

    if (!a_DataFile)
    {
        ostringstream tStream;
        tStream << "NN file error!" << std::endl;
        //    throw NS::Exception(tStream.str());
    }

    // search for NNstart
    do
    {
        a_DataFile >> t_str;
        if (t_str == "NNstart")
            t_no_start = false;

    }
    while ((t_str != "NNstart") && (!a_DataFile.eof()));

    if (t_no_start)
        return false;

    Clear();

    // read in the input/output dimentions
    a_DataFile >> m_num_inputs;
    a_DataFile >> m_num_outputs;

    // read in all data
    do
    {
        a_DataFile >> t_str;

        // a neuron?
        if (t_str == "neuron")
        {
            Neuron t_n;

            // for type and aftype
            int t_type, t_aftype;

            a_DataFile >> t_type;
            a_DataFile >> t_n.m_a;
            a_DataFile >> t_n.m_b;
            a_DataFile >> t_n.m_timeconst;
            a_DataFile >> t_n.m_bias;
            a_DataFile >> t_aftype;
            a_DataFile >> t_n.m_split_y;
            a_DataFile >> t_n.id;

            t_n.m_type = static_cast<NEAT::NeuronType>(t_type);
            t_n.m_activation_function_type = static_cast<NEAT::ActivationFunction>(t_aftype);

            m_neurons[t_n.id] = t_n;
        }

        // a connection?
        if (t_str == "connection")
        {
            Connection t_c;

            int t_isrecur;

            a_DataFile >> t_c.m_source_neuron_idx;
            a_DataFile >> t_c.m_target_neuron_idx;
            a_DataFile >> t_c.m_weight;
            a_DataFile >> t_isrecur;

            a_DataFile >> t_c.m_hebb_rate;
            a_DataFile >> t_c.m_hebb_pre_rate;

            t_c.m_recur_flag = static_cast<bool>(t_isrecur);

            m_connections.push_back(t_c);
        }



        if (t_str == "NNend")
            t_no_end = false;
    }
    while ((t_str != "NNend") && (!a_DataFile.eof()));

    if (t_no_end)
    {
        ostringstream tStream;
        tStream << "NNend not found in file!" << std::endl;
        //    throw NS::Exception(tStream.str());
    }

    return true;
}
bool NeuralNetwork::Load(const char *a_filename)
{
    std::ifstream t_DataFile(a_filename);
    return Load(t_DataFile);
}

unsigned int NeuralNetwork::CalculateDepth()
{
    if (is_depth_ready){
        return m_Depth;
    }

    unsigned int t_max_depth = 0;
    unsigned int t_cur_depth = 0;

    // The quick case - if no hidden neurons,
    // the depth is 1
    if (NumNeurons() == NumInputs() + NumOutputs())
    {
        m_Depth = 1;
        return m_Depth;
    }

    // make a list of all output IDs
    std::vector<unsigned int> t_output_ids;
    for(unsigned int i=0; i < NumNeurons(); i++)
    {
        if (m_neurons[i].Type() == OUTPUT)
        {
            t_output_ids.push_back(i);
        }
    }

    // For each output
    for(unsigned int i=0; i<t_output_ids.size(); i++)
    {
        t_cur_depth = NeuronDepth(t_output_ids[i], 0);

        if (t_cur_depth > t_max_depth)
            t_max_depth = t_cur_depth;
    }

    m_Depth = t_max_depth;
    is_depth_ready = true;
    return m_Depth;
}


unsigned int NeuralNetwork::NeuronDepth(unsigned int a_NeuronID, unsigned int a_Depth)
{
    unsigned int t_max_depth = a_Depth;

    if (a_Depth > 16)
    {
        // oops! a loop in the network!
        //std::cout << std::endl << " ERROR! Trying to get the depth of a looped network!" << std::endl;
        return 16;
    }

    // Base case
    if ((GetNeuronByID(a_NeuronID).Type() == INPUT) || (GetNeuronByID(a_NeuronID).Type() == BIAS))
    {
        return a_Depth;
    }

    // Find all links outputting to this neuron ID
    auto t_inputting_links_idx = m_connections.get<ConnectionTargetMap>().equal_range(a_NeuronID);

    // For all incoming links..
    for (auto c_it=t_inputting_links_idx.first; c_it != t_inputting_links_idx.second; c_it++)
    {
        const Connection & t_link = *c_it;
        // RECURSION
        unsigned int t_current_depth = NeuronDepth(t_link.m_source_neuron_idx, a_Depth + 1);
        if (t_current_depth > t_max_depth)
            t_max_depth = t_current_depth;
        if (t_max_depth >= 16)
            return 16;
    }

    return t_max_depth;
}

void NeuralNetwork::AddNeuron(const Neuron a_n) {
    m_neurons.push_back(a_n) ;
    this->_activated.push_back(false);
    this->_inActivation.push_back(false);
}

void NeuralNetwork::RecursiveActivation(){
    // make a list of all output IDs
    std::vector<unsigned int> t_output_ids;
    std::vector<unsigned int> t_input_ids;
    for(unsigned int i=0; i < NumNeurons(); i++)
    {
        if (m_neurons[i].Type() == OUTPUT)
        {
            t_output_ids.push_back(i);
        }
        if (m_neurons[i].Type() == INPUT)
        {
            t_input_ids.push_back(i);
        }

    }

   // Initialize boolean arrays and set the last activation signal, but only if it isn't an input (these have already been set when the input is activated)
   for (uint i = 0; i < this->NumNeurons(); i++)
   {
        if (m_neurons[i].Type() == INPUT || m_neurons[i].Type() == BIAS)
            // Set as activated if i is an input node, otherwise ensure it is unactivated (false)
            _activated[i] = true;
        else
            _activated[i] = false;
       this->_inActivation[i] = false;
   }

   // Get each output node activation recursively
   for(uint i=0; i<t_output_ids.size(); i++)
       RecursiveActivateNode(t_output_ids[i]);
}

void NeuralNetwork::RecursiveActivateNode(int a_NeuronID)
{
   // If we've reached an input node we return since the signal is already set
   if (this->_activated[a_NeuronID])
   {
       this->_inActivation[a_NeuronID] = false;
       return;
   }

   // Mark that the node is currently being calculated
   this->_inActivation[a_NeuronID] = true;


   // Find all links outputting to this neuron ID
   auto t_inputting_links_idx = m_connections.get<ConnectionTargetMap>().equal_range(a_NeuronID);


   // Adjacency list in reverse holds incoming connections, go through each one and activate it
   for (auto c_it=t_inputting_links_idx.first; c_it!=t_inputting_links_idx.second; c_it++)
   {
       //{ Region recurrant connection handling - not applicable in our implementation
       // If this node is currently being activated then we have reached a cycle, or recurrant connection. Use the previous activation in this case
       //m_source_neuron_idx
       //m_target_neuron_idx
       if (!(this->_inActivation[c_it->m_source_neuron_idx]))
       /*
       {
           c_it->neuronSignalsBeingProcessed[currentNode] += this.lastActivation[crntAdjNode] * this.adjacentMatrix[crntAdjNode, currentNode];
       }*/

       // Otherwise proceed as normal
       //else
       {
           // Recurse if this neuron has not been activated yet
           if (!this->_activated[c_it->m_source_neuron_idx]){
               RecursiveActivateNode(c_it->m_source_neuron_idx);
            }
           // Add it to the new activation
           this->m_neurons[a_NeuronID].m_activesum += m_neurons[c_it->m_source_neuron_idx].m_activation * c_it->m_weight;
       }
       //calculate activation


       //} endregion
   }
   // Set this signal after running it through the activation function

    CalculateNeuronActivation(a_NeuronID);

   // Mark this neuron as completed
   this->_activated[a_NeuronID] = true;

   // This is no longer being calculated (for cycle detection)
   this->_inActivation[a_NeuronID] = false;

}


Neuron & NeuralNetwork::GetNeuronByID(uint ID){
    return this->m_neurons[ID];
}

} // namespace NEAT
