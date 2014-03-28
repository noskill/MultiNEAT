#ifndef EVOLVABLESUBSTRATE_H
#define EVOLVABLESUBSTRATE_H
#include <deque>
#include <map>
#include <memory>
#include <math.h>
#include "Genes.h"


class NeuralNetwork;

namespace NEAT
{

template<typename T>
struct Point{

    T X;
    T Y;
    Point(T x, T y):
        X(x), T(y){}
};

typedef Point<float> PointF;
typedef Point<double> PointD;

class EvolvableSubstrate
{
private:
    std::map<PointD> __hiddenInsertIndex;
    std::map<PointD> __inputInsertIndex;
    std::map<PointD> __outputInsertIndex;
public:

    std::vector<PointD> inputCoordinates;
    std::vector<PointD> hiddenCoordinates;
    std::vector<PointD> outputCoordinates;

    std::vector<LinkGene> m_links;

    float divisionThreshold;
    float varianceThreshold;
    float bandThrehold;
    NeuralNetwork * cppn;

    struct QuadPoint
    {
        float x, y;
        float w; //stores the CPPN value
        float width; //width of this quadtree square
        std::vector<QuadPoint> childs;
        int level; //the level in the quadtree

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
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.weight = weight;
        }
    };

private:
    Parameters parameters;

    /*
     * Input: Coordinates of source (outgoing = true) or target node (outgoing = false) at (a,b)
     * Output: Quadtree, in which each quadnode at (x,y) stores CPPN activation level for its
     *         position. The initialized quadtree is used in the PruningAndExtraction phase to
     *         generate the actual ANN connections.
     */
    QuadPoint QuadTreeInitialisation(float a, float b, bool outgoing, int initialDepth, int maxDepth);

    /*
     * Input : Coordinates of source (outgoing = true) or target node (outgoing = false) at (a,b) and initialized quadtree p->
     * Output: Adds the connections that are in bands of the two-dimensional cross-section of the
     *         hypercube containing the source or target node to the connections list.
     *
     */

    std::vector<TempConnection> PruneAndExpress(float a, float b, QuadPoint & node, bool outgoing, float maxDepth);

    //Collect the CPPN values stored in a given quadtree p
    //Used to estimate the variance in a certain region in space

    void getCPPNValues(std::vector<float> & l, QuadPoint & p);

public:

    EvolvableSubstrate(Parameters & p, py::list a_inputs, py::list a_outputs);

    void setCPPN(NeuralNetwork * cppn);

    //determine the variance of a certain region
    float variance(QuadPoint & p)
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
        m /= l.Count;
        for (float f: l)
        {
            v += (float)((f - m)^2);
        }
        v /= l.Count;
        return v;
    }

    float queryCPPN(float x1, float y1, float x2, float y2);

    /*
     * The main method that generations a list of ANN connections based on the information in the
     * underlying hypercube.
     * Input : CPPN, InputPositions, OutputPositions, ES-HyperNEAT parameters
     * Output: Connections, HiddenNodes
     */
    void generateSubstrate(NeuralNetwork & cppn, int initialDepth, float varianceThreshold, float bandThreshold, int ESIterations,
                                            float divsionThreshold, int maxDepth,
                                            unsigned int inputCount, unsigned int outputCount);
};

}

#endif // EVOLVABLESUBSTRATE_H
