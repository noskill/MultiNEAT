#include "EvolvableSubstrate.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Genes.h"



using namespace NEAT;

EvolvableSubstrate::EvolvableSubstrate(Parameters const & p, boost::python::list const &a_inputs, boost::python::list const &a_outputs):
    parameters(p)
{


    // Make room for the data
    uint inp = py::len(a_inputs);
    uint out = py::len(a_outputs);
    this->inputCount = inp;
    this->outputCount = out;

    Coordinates.reserve( out + inp );

    for(uint i=0; i<inp; i++)
    {
        std::vector<double> i_input;
        for(uint j=0; j<py::len(a_inputs[i]); j++){
            i_input.push_back(py::extract<double>(a_inputs[i][j]));
        }
        Coordinates.emplace_back(i_input);
        __InsertIndex[Coordinates.back()] = i;
    }

    assert((Coordinates.size() == inp) && (__InsertIndex.size() == inp));

    for(uint i=0; i<out; i++)
    {
        std::vector<double> i_out;
        for(uint j=0; j<py::len(a_outputs[i]); j++){
            i_out.push_back(py::extract<double>(a_outputs[i][j]));
        }
        Coordinates.emplace_back(i_out);
        __InsertIndex[Coordinates.back()] = i + inp;
    }

    assert((Coordinates.size() == inp + out) && (__InsertIndex.size() == inp + out));
}

EvolvableSubstrate::QuadPoint EvolvableSubstrate::QuadTreeInitialisation(float a, float b, bool outgoing){
    QuadPoint root = QuadPoint(0.0f, 0.0f, 1.0f, 1); //x, y, width, level
    auto queue = std::deque<QuadPoint*>();
    queue.push_back(&root);

    while (!queue.empty())
    {
        QuadPoint * p = queue.front();//dequeue
        queue.pop_front();

        // Divide into sub-regions and assign children to parent
        p->childs.push_back(QuadPoint(p->x - p->width / 2, p->y - p->width / 2, p->width / 2, p->level + 1));
        p->childs.push_back(QuadPoint(p->x - p->width / 2, p->y + p->width / 2, p->width / 2, p->level + 1));
        p->childs.push_back(QuadPoint(p->x + p->width / 2, p->y - p->width / 2, p->width / 2, p->level + 1));
        p->childs.push_back(QuadPoint(p->x + p->width / 2, p->y + p->width / 2, p->width / 2, p->level + 1));

        for (auto & c : p->childs)
        {
            if (outgoing) // Querying connection from input or hidden node
            {
                c.w = queryCPPN(a, b, c.x, c.y); // Outgoing connectivity pattern
            }
            else // Querying connection to output node
            {
                c.w = queryCPPN(c.x, c.y, a, b); // Incoming connectivity pattern
            }
        }

        // Divide until initial resolution or if variance is still high
        if (p->level < parameters.InitialDepth || (p->level < parameters.MaximumDepth && variance((*p)) > parameters.DivisionThreshold))
        {
            for (QuadPoint & c : p->childs)
            {
                queue.push_back(&c);
            }
        }
    }
    return root;
}

void EvolvableSubstrate::setCPPN(NeuralNetwork * net){
    this->cppn = net;
}

std::vector<EvolvableSubstrate::TempConnection> EvolvableSubstrate::PruneAndExpress(float a, float b, QuadPoint & node, bool outgoing)
{
    std::vector<TempConnection> temp_connections;

    float left = 0.0f, right = 0.0f, top = 0.0f, bottom = 0.0f;

    if (node.childs.size() == 0) return temp_connections;

    // Traverse quadtree depth-first
    for (QuadPoint & c : node.childs)
    {
        float childVariance = variance(c);

        if (childVariance >= parameters.VarianceThreshold)
        {
            for(auto & _tmp_conn: PruneAndExpress(a, b, c, outgoing)){
                temp_connections.push_back(_tmp_conn);
            }
        }
        else //this should always happen for at least the leaf nodes because their variance is zero
        {
            // Determine if point is in a band by checking neighbor CPPN values
            if (outgoing)
            {
                left = std::abs(c.w - queryCPPN(a, b, c.x - node.width, c.y));
                right = std::abs(c.w - queryCPPN(a, b, c.x + node.width, c.y));
                top = std::abs(c.w - queryCPPN(a, b, c.x, c.y - node.width));
                bottom = std::abs(c.w - queryCPPN(a, b, c.x, c.y + node.width));
            }
            else
            {
                left = std::abs(c.w - queryCPPN(c.x - node.width, c.y, a, b));
                right = std::abs(c.w - queryCPPN(c.x + node.width, c.y, a, b));
                top = std::abs(c.w - queryCPPN(c.x, c.y - node.width, a, b));
                bottom = std::abs(c.w - queryCPPN(c.x, c.y + node.width, a, b));
            }

            if (std::max(std::min(top, bottom), std::min(left, right)) > parameters.BandingThreshold)
            {
                if (outgoing)
                {
                    temp_connections.emplace_back(a, b, c.x, c.y, c.w);
                }
                else
                {
                    temp_connections.emplace_back(c.x, c.y, a, b, c.w);
                }
            }

        }
    }

    return temp_connections;
}

void EvolvableSubstrate::clearHidden(){
    this->m_connections.clear();
    this->Coordinates.clear();
    this->__InsertIndex.clear();
}

void EvolvableSubstrate::generateSubstrate(NeuralNetwork _cppn)
{
    setCPPN(&_cppn);
    this->clearHidden();

    std::vector<TempConnection> tempConnections;
    uint innovationCounter = 0;

    //CONNECTIONS DIRECTLY FROM INPUT NODES
    for (uint sourceIndex=0; sourceIndex < inputCount; sourceIndex++)
    {
        PointD & input = Coordinates[sourceIndex];

        // Analyze outgoing connectivity pattern from this input
        QuadPoint root = QuadTreeInitialisation(input.X, input.Y, true);

        // Traverse quadtree and add connections to list
        tempConnections = PruneAndExpress(input.X, input.Y, root, true);

        for (TempConnection & p : tempConnections)
        {
            PointD newp = PointD(p.x2, p.y2);
            uint targetIndex;
            auto it = __InsertIndex.find(newp);
            if (it == __InsertIndex.end())
            {

                targetIndex = Coordinates.size();
                Coordinates.push_back(newp);
                __InsertIndex[newp] = targetIndex;
            }
            else{
                targetIndex = it->second;
            }
            m_connections.push_back(LinkGene(sourceIndex, targetIndex, innovationCounter++, p.weight));
        }
    }

    tempConnections.clear();

    //HIDDEN TO HIDDEN NEURONS
    std::map<PointD, uint> unexploredHiddenNodes = getHiddenPoints();

    for (uint step = 0; step < this->parameters.ESIterations; step++)
    {
        for (auto & hiddenP: unexploredHiddenNodes)
        {
            QuadPoint root = QuadTreeInitialisation(hiddenP.first.X, hiddenP.first.Y, true);
            tempConnections = PruneAndExpress(hiddenP.first.X, hiddenP.first.Y, root, true);

            uint sourceIndex = hiddenP.second;

            for (TempConnection const & p: tempConnections)
            {

                PointD newp = PointD(p.x2, p.y2);

                auto it = __InsertIndex.find(newp);
                size_t targetIndex;
                if (it == __InsertIndex.end())
                {
                    targetIndex = Coordinates.size();
                    Coordinates.push_back(newp);
                    __InsertIndex[newp] = targetIndex;
                }
                else{
                    targetIndex = it->second;
                }
                m_connections.emplace_back(sourceIndex, targetIndex, innovationCounter++, p.weight);
            }
        }

        std::map<PointD, uint> temp = getHiddenPoints();

        // Remove the just explored nodes
        for (auto f: unexploredHiddenNodes){
            temp.erase(f.first);
        }

        unexploredHiddenNodes = std::move(temp);

    }

    tempConnections.clear();

    // CONNECT HIDDEN TO OUTPUT
    for (uint targetIndex = inputCount; targetIndex < inputCount + outputCount; targetIndex++)
    {
        const PointD & outputPos = Coordinates[targetIndex];

        // Analyze incoming connectivity pattern to this output
        QuadPoint root = QuadTreeInitialisation(outputPos.X, outputPos.Y, false);

        tempConnections = PruneAndExpress(outputPos.X, outputPos.Y, root, false);

        for (const TempConnection &  t: tempConnections)
        {
            PointD source(t.x1, t.y1);
            auto it = __InsertIndex.find(source);

            /*
            New nodes not created here because all the hidden nodes that are
                connected to an input/hidden node are already expressed.
            */
            if (it != __InsertIndex.end()){  //only connect if hidden neuron already exists
                uint sourceIndex = it->second;
                m_connections.push_back(LinkGene(sourceIndex, targetIndex, innovationCounter++, t.weight));
            }
        }

    }
    assert([this]() {
       for(LinkGene & l: this->m_connections){
           assert(l.FromNeuronID() < 100);
           assert(l.ToNeuronID() < 100);
       }
       return true;
    }());
    tempConnections.clear();
}

double EvolvableSubstrate::queryCPPN(float x1, float y1, float x2, float y2)
{
    std::vector<double> coordinates;
    coordinates.reserve(5);
    coordinates.push_back(x1);
    coordinates.push_back(y1);
    coordinates.push_back(x2);
    coordinates.push_back(y2);
    coordinates.push_back(1.0);  // bias

    cppn->Input(coordinates);
    cppn->RecursiveActivation();
    std::vector<double> output = cppn->Output();
    ASSERT(output.size() == 1);
    return output[0];
}

void EvolvableSubstrate::getCPPNValues(std::vector<float> & l, const QuadPoint & p)
{
    if (!p.childs.empty())
    {
        for (int i = 0; i < 4; i++)
        {
            getCPPNValues(l, p.childs[i]);
        }
    }
    else
    {
        l.push_back(p.w);
    }
}

std::map<PointD, uint> EvolvableSubstrate::getHiddenPoints(){
    std::map<PointD, uint> result;
    for(size_t i=inputCount + outputCount; i < Coordinates.size(); i++){
        result[Coordinates[i]] = i + inputCount + outputCount;
    }
    return result;
}

//debug and test functions

void EvolvableSubstrate::checkConnections(){
    for(size_t i=0; i<this->m_connections.size(); i++){
        assert(pointExists(m_connections[i].FromNeuronID()));
        assert(pointExists(m_connections[i].ToNeuronID()));
    }
}


bool EvolvableSubstrate::pointExists(size_t neuronId){
    bool result = false;
    if (__InsertIndex.find(Coordinates[neuronId]) != __InsertIndex.end()){
        result = true;
    }
    return result;
}
