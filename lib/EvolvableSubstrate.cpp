#include "EvolvableSubstrate.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Genes.h"



using namespace NEAT;

EvolvableSubstrate::EvolvableSubstrate(Parameters const & p, boost::python::list const &a_inputs, boost::python::list const &a_outputs):
    hiddenInsertIndex(0.0, 0.0, 1, 0),
    outputInsertIndex(0.0, 0.0, 1, 0),
    inputInsertIndex(0.0, 0.0, 1, 0),
    parameters(p)
{


    // Make room for the data
    uint inp = py::len(a_inputs);
    uint out = py::len(a_outputs);
    this->inputCount = inp;
    this->outputCount = out;

    for(uint i=0; i<inp; i++)
    {
        std::vector<double> i_input;
        for(uint j=0; j<py::len(a_inputs[i]); j++){
            i_input.push_back(py::extract<double>(a_inputs[i][j]));
        }
        inputInsertIndex[PointD(i_input)] = i;
    }

    assert(inputInsertIndex.size() == inp);

    for(uint i=0; i<out; i++)
    {
        std::vector<double> i_out;
        for(uint j=0; j<py::len(a_outputs[i]); j++){
            i_out.push_back(py::extract<double>(a_outputs[i][j]));
        }
        outputInsertIndex[PointD(i_out)] = i + inp;
    }

    assert(outputInsertIndex.size() == out);
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
                left = std::fabs(c.w - queryCPPN(a, b, c.x - node.width, c.y));
                right = std::fabs(c.w - queryCPPN(a, b, c.x + node.width, c.y));
                top = std::fabs(c.w - queryCPPN(a, b, c.x, c.y - node.width));
                bottom = std::fabs(c.w - queryCPPN(a, b, c.x, c.y + node.width));
            }
            else
            {
                left = std::fabs(c.w - queryCPPN(c.x - node.width, c.y, a, b));
                right = std::fabs(c.w - queryCPPN(c.x + node.width, c.y, a, b));
                top = std::fabs(c.w - queryCPPN(c.x, c.y - node.width, a, b));
                bottom = std::fabs(c.w - queryCPPN(c.x, c.y + node.width, a, b));
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
    hiddenInsertIndex.clear();
    this->m_connections.clear();
}

void EvolvableSubstrate::generateSubstrate(NeuralNetwork _cppn)
{
    setCPPN(&_cppn);
    this->clearHidden();
    assert(hiddenInsertIndex.checkSize());
    std::vector<TempConnection> tempConnections;
    uint innovationCounter = 0;

    //CONNECTIONS DIRECTLY FROM INPUT NODES
    for(auto it=inputInsertIndex.begin(); it!=inputInsertIndex.end();it++)
    {
        const PointD & input = *it;
        const uint sourceIndex = (*it).data;
        // Analyze outgoing connectivity pattern from this input
        QuadPoint root = QuadTreeInitialisation(input.X, input.Y, true);

        // Traverse quadtree and add connections to list
        tempConnections = PruneAndExpress(input.X, input.Y, root, true);

        for (TempConnection & p : tempConnections)
        {
            PointD newp = PointD(p.x2, p.y2);
            uint targetIndex;
            //new point is not an input point
            if(!existInput(newp))
            {
                auto hidden_it = hiddenInsertIndex.find(newp);
                if (hidden_it == hiddenInsertIndex.end())
                {
                    targetIndex = hiddenInsertIndex.size() + inputCount + outputCount;
                    hiddenInsertIndex[newp] = targetIndex;

                }
                else{
                    targetIndex = (*hidden_it).data;
                }
                if(targetIndex!=sourceIndex){
                m_connections.push_back(LinkGene(sourceIndex, targetIndex, innovationCounter++, p.weight));
                }
            }
        }
    }

    tempConnections.clear();

    //HIDDEN TO HIDDEN NEURONS
    SQuadPoint<PointD> unexploredHiddenNodes = getHiddenPoints();

    for (uint step = 0; step < this->parameters.ESIterations; step++)
    {
        for (auto & hiddenP: unexploredHiddenNodes)
        {
            QuadPoint root = QuadTreeInitialisation(hiddenP.X, hiddenP.Y, true);

            tempConnections = PruneAndExpress(hiddenP.X, hiddenP.Y, root, true);

            const uint sourceIndex = hiddenP.data;

            for (TempConnection const & p: tempConnections)
            {
                PointD newp = PointD(p.x2, p.y2);

                if(!existNonHidden(newp)){
                    auto it = hiddenInsertIndex.find(newp);
                    size_t targetIndex;
                    if (it == hiddenInsertIndex.end())
                    {
                        targetIndex = hiddenInsertIndex.size()  + inputCount + outputCount;;
                        hiddenInsertIndex[newp] = targetIndex;
                    }
                    else{
                        targetIndex = (*it).data;
                    }

                    if(targetIndex!=sourceIndex){
                        if ((step == 0) ? true : (connectionCount(sourceIndex, targetIndex) == 0)){
                            m_connections.emplace_back(sourceIndex, targetIndex, innovationCounter++, p.weight);
                        }
                    }
                }
            }
        }

        SQuadPoint<PointD> temp = getHiddenPoints();

        // Remove the just explored nodes
        for (auto f: unexploredHiddenNodes){
            temp.erase(f);
            assert(hiddenInsertIndex.find(f) != hiddenInsertIndex.end());
        }

        unexploredHiddenNodes = std::move(temp);

    }

    tempConnections.clear();

    // CONNECT HIDDEN TO OUTPUT
    uint _vv=0;
    for(auto it=outputInsertIndex.begin(); it!=outputInsertIndex.end();it++)
    {

        const PointD & outputPos = *it;
        const uint targetIndex = (*it).data;

        // Analyze incoming connectivity pattern to this output
        QuadPoint root = QuadTreeInitialisation(outputPos.X, outputPos.Y, false);

        tempConnections = PruneAndExpress(outputPos.X, outputPos.Y, root, false);
        _vv++;

        for (const TempConnection &  t: tempConnections)
        {
            PointD source(t.x1, t.y1);

            auto hidden_it = hiddenInsertIndex.find(source);

            /*
            New nodes not created here because all the hidden nodes that are
                connected to an input/hidden node are already expressed.
            */
            if (hidden_it != outputInsertIndex.end()){  //only connect if hidden neuron already exists
                uint sourceIndex = (*hidden_it).data;
                if(targetIndex!=sourceIndex){
                    m_connections.push_back(LinkGene(sourceIndex, targetIndex, innovationCounter++, t.weight));
                    assert(checkNeuronID(sourceIndex).size() == 1);
                }
            }
        }
    }

    tempConnections.clear();
    assert(checkConnections());
    assert(hiddenInsertIndex.checkSize());
}

bool EvolvableSubstrate::existNonHidden(PointD & p){
    auto out = outputInsertIndex.find(p);
    auto inp = inputInsertIndex.find(p);
    bool result = true;
    if(inp == inputInsertIndex.end() && out == outputInsertIndex.end()){
        result = false;
    }
    return result;
}

bool EvolvableSubstrate::existInput(PointD & p){
    auto inp = inputInsertIndex.find(p);
    bool result = true;
    if(inp == inputInsertIndex.end()){
        result = false;
    }
    return result;
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

SQuadPoint<PointD> EvolvableSubstrate::getHiddenPoints(){
    SQuadPoint<PointD> result = this->hiddenInsertIndex;
    return result;
}

//debug and test functions

bool EvolvableSubstrate::checkConnections(){
    for(size_t i=0; i<this->m_connections.size(); i++){
        assert(pointExists(m_connections[i].FromNeuronID()));
        assert(pointExists(m_connections[i].ToNeuronID()));
        assert(connectionCount(m_connections[i].FromNeuronID(), m_connections[i].ToNeuronID()) == 1);
    }
    return true;
}

std::vector<PointD> EvolvableSubstrate::checkNeuronID(uint a_id){
    std::vector<PointD> result;
    std::vector< SQuadPoint<PointD> * > vec = {&inputInsertIndex, &outputInsertIndex, &hiddenInsertIndex};
    for(auto & InsertIndex: vec){
        for(auto it=InsertIndex->begin(); it!=InsertIndex->end();it++){
            if((*it).data == a_id){
                result.push_back(*it);
            }
        }
    }
    assert(result.size() <= 1);
    return result;
}

uint EvolvableSubstrate::connectionCount(uint source, uint target){
    std::vector<uint> result;
    for(size_t i=0; i<this->m_connections.size(); i++){
        if(m_connections[i].FromNeuronID() == source && m_connections[i].ToNeuronID() == target){
            result.push_back(i);
        }
        if(result.size()>1)
            assert(false);
    }
    return result.size();
}

bool EvolvableSubstrate::pointExists(size_t neuronId){
    bool result = false;
    std::vector< SQuadPoint<PointD> * > vec = {&inputInsertIndex, &outputInsertIndex, &hiddenInsertIndex};
    for(auto & InsertIndex: vec){
        for(auto it=InsertIndex->begin(); it!=InsertIndex->end() && !result;it++){
            if((*it).data == neuronId){
                result = true;
            }
        }
    }

    return result;
}

size_t EvolvableSubstrate::CoordinatesSize(){
    return this->inputInsertIndex.size() + this->outputInsertIndex.size() + this->hiddenInsertIndex.size();
}
