#ifndef EVOLVABLESUBSTRATE_H
#define EVOLVABLESUBSTRATE_H
#include <deque>
#include <map>
#include <memory>
#include <math.h>
#include "Genes.h"
#include "Assert.h"
#include "Point.h"
#include <algorithm>
#include "SubstrateBase.h"


namespace NEAT
{

class NeuralNetwork;


class EvolvableSubstrate: public BaseSubstrate
{
private:
    std::map<PointD, uint> __hiddenInsertIndex;
    std::map<PointD, uint> __inputInsertIndex;
    std::map<PointD, uint> __outputInsertIndex;

public:
 //   bool m_leaky = false;
 //   bool m_with_distance = false;
    NEAT::ActivationFunction m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    NEAT::ActivationFunction m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    /*
    bool m_allow_input_hidden_links = true;
    bool m_allow_input_output_links = true;
    bool m_allow_hidden_hidden_links = true;
    bool m_allow_hidden_output_links = true;
    bool m_allow_output_hidden_links = true;
    bool m_allow_output_output_links = true;
    bool m_allow_looped_hidden_links = true;
    bool m_allow_looped_output_links = true;
    float m_link_threshold = 0.2;
    float m_max_weight_and_bias = 5.0;
    float m_min_time_const = 0.1;
    float m_max_time_const = 1.0;*/

    std::vector<PointD> inputCoordinates;
    std::vector<PointD> hiddenCoordinates;
    std::vector<PointD> outputCoordinates;
    std::vector<LinkGene> connections;

    NeuralNetwork * cppn;

    struct QuadPoint
    {
        float x, y;
        float w; //stores the CPPN value
        float width; //width of this quadtree square
        std::vector<QuadPoint> childs;
        uint level; //the level in the quadtree

        QuadPoint(float _x, float _y, float _w, int _level)
        {
            level = _level;
            w = 0.0f;
            x = _x;
            y = _y;
            width = _w;
            childs = std::vector<QuadPoint>();
        }
    };

    struct TempConnection
    {
        float x1, y1, x2, y2;
        //public PointF start, end;
        float weight;
        TempConnection(float x1, float y1, float x2, float y2, float weight)
        {
        //    start = new PointF(x1, y1);
            this->x1 = x1;
            this->y1 = y1;
            this->x2 = x2;
            this->y2 = y2;
            this->weight = weight;
        }
    };

private:

    const Parameters & parameters;

    /*
     * Input: Coordinates of source (outgoing = true) or target node (outgoing = false) at (a,b)
     * Output: Quadtree, in which each quadnode at (x,y) stores CPPN activation level for its
     *         position. The initialized quadtree is used in the PruningAndExtraction phase to
     *         generate the actual ANN connections.
     */
    QuadPoint QuadTreeInitialisation(float a, float b, bool outgoing);

    /*
     * Input : Coordinates of source (outgoing = true) or target node (outgoing = false) at (a,b) and initialized quadtree p->
     * Output: Adds the connections that are in bands of the two-dimensional cross-section of the
     *         hypercube containing the source or target node to the connections list.
     *
     */

    std::vector<TempConnection> PruneAndExpress(float a, float b, QuadPoint & node, bool outgoing);

    //Collect the CPPN values stored in a given quadtree p
    //Used to estimate the variance in a certain region in space

    void getCPPNValues(std::vector<float> & l, const QuadPoint &p);

    void clearHidden();

public:

    EvolvableSubstrate(Parameters const & p, py::list const & a_inputs, py::list const & a_outputs);

    void setCPPN(NeuralNetwork * cppn);

    //determine the variance of a certain region
    float variance(const QuadPoint & p)
    {
        if (p.childs.empty())
        {
            return 0.0f;
        }

        std::vector<float> l =  std::vector<float>();
        getCPPNValues(l, p);

        float m = 0.0f, v = 0.0f;
        for (float f: l)
        {
            m += f;
        }
        m /= l.size();
        for (float f: l)
        {
            v += (float)(std::pow(f - m, 2));
        }
        v /= l.size();
        return v;
    }

    float queryCPPN(float x1, float y1, float x2, float y2);

    /*
     * The main method that generations a list of ANN connections based on the information in the
     * underlying hypercube.
     * Input : CPPN, InputPositions, OutputPositions, ES-HyperNEAT parameters
     * Output: Connections, HiddenNodes
     */
    void generateSubstrate(NeuralNetwork cppn);

    //dementionality of substrate
    uint GetMaxDims(){
        uint result = 0;
        if (this->inputCoordinates.size())
            result = std::max((uint)this->inputCoordinates[0].size(), result);
        if (this->hiddenCoordinates.size())
            result = std::max((uint)this->hiddenCoordinates[0].size(), result);
        if (this->outputCoordinates.size())
            result = std::max((uint)this->outputCoordinates[0].size(), result);
        return result;
    }

    uint GetMinCPPNInputs(){
        return PointD::SIZE;
    }

    uint GetMinCPPNOutputs(){
        return 1;
    }
};

}

#endif // EVOLVABLESUBSTRATE_H
