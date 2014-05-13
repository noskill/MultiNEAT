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


class EvolvableSubstrate: public SubstrateBase
{
private:
    EvolvableSubstrate()=delete;

public:
    std::map<PointD, uint> hiddenInsertIndex;
    std::map<PointD, uint> outputInsertIndex;
    std::map<PointD, uint> inputInsertIndex;
    NEAT::ActivationFunction m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    NEAT::ActivationFunction m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    std::vector<LinkGene> m_connections;
    uint inputCount;
    uint outputCount;

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
        TempConnection(float _x1, float _y1, float _x2, float _y2, float _weight):
        x1 (_x1), y1 (_y1), x2(_x2), y2(_y2), weight(_weight)
        {
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
    bool existNonHidden(PointD & p);

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

    double queryCPPN(float x1, float y1, float x2, float y2);

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
        for(auto it=inputInsertIndex.begin(); it!=inputInsertIndex.end(); it++){
            result = std::max((uint)(it->first.size()), result);
        }
        for(auto it=outputInsertIndex.begin(); it!=outputInsertIndex.end(); it++){
            result = std::max((uint)(it->first.size()), result);
        }
        for(auto it=hiddenInsertIndex.begin(); it!=hiddenInsertIndex.end(); it++){
            result = std::max((uint)(it->first.size()), result);
        }
        return result;
    }

    std::map<PointD, uint> getHiddenPoints();

    //debug and test functions

    bool checkConnections();

    bool existInput(PointD & p);

    uint connectionCount(uint source, uint target);

    bool pointExists(size_t neuronId);

    size_t CoordinatesSize();
};

}

#endif // EVOLVABLESUBSTRATE_H
