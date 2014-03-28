#include "EvolvableSubstrate.h"
#include "NeuralNetwork.h"
#include "Parameters.h"





EvolvableSubstrate::EvolvableSubstrate(Parameters & p, py::list a_inputs, py::list a_outputs):
    paramters(p)
{
    m_leaky = false;
    m_with_distance = false;
    m_hidden_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_output_nodes_activation = NEAT::UNSIGNED_SIGMOID;
    m_allow_input_hidden_links = true;
    m_allow_input_output_links = true;
    m_allow_hidden_hidden_links = true;
    m_allow_hidden_output_links = true;
    m_allow_output_hidden_links = true;
    m_allow_output_output_links = true;
    m_allow_looped_hidden_links = true;
    m_allow_looped_output_links = true;

    m_link_threshold = 0.2;
    m_max_weight_and_bias = 5.0;
    m_min_time_const = 0.1;
    m_max_time_const = 1.0;

    // Make room for the data
    int inp = py::len(a_inputs);
    int out = py::len(a_outputs);

    m_input_coords.resize( inp );
    hiddenCoordinates.resize( hid );
    m_output_coords.resize( out );

    for(int i=0; i<inp; i++)
    {
        for(int j=0; j<py::len(a_inputs[i]); j++)
            m_input_coords[i].push_back(py::extract<double>(a_inputs[i][j]));
    }
    for(int i=0; i<hid; i++)
    {
        for(int j=0; j<py::len(a_hidden[i]); j++)
            hiddenCoordinates[i].push_back(py::extract<double>(a_hidden[i][j]));
    }
    for(int i=0; i<out; i++)
    {
        for(int j=0; j<py::len(a_outputs[i]); j++)
            m_output_coords[i].push_back(py::extract<double>(a_outputs[i][j]));
    }
}

QuadPoint EvolvableSubstrate::QuadTreeInitialisation(float a, float b, bool outgoing, int initialDepth, int maxDepth){
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
        if (p->level < initialDepth || (p->level < maxDepth && variance(p) > divisionThreshold))
        {
            for (QuadPoint & c : p->childs)
            {
                queue.push_back(*c);
            }
        }
    }
    return root;
}

void EvolvableSubstrate::setCPPN(NeuralNetwork * net){
    this.cppn = net;
}

std::vector<TempConnection> EvolvableSubstrate::PruneAndExpress(float a, float b, QuadPoint & node, bool outgoing, float maxDepth)
{
    std::vector<TempConnection> connections;

    float left = 0.0f, right = 0.0f, top = 0.0f, bottom = 0.0f;

    if (!node.childs[0]) return;

    // Traverse quadtree depth-first
    for (QuadPoint & c : node.childs)
    {
        float childVariance = variance(c);

        if (childVariance >= varianceThreshold)
        {
            for(auto & _tmp_conn: PruneAndExpress(a, b, connections, c, outgoing, maxDepth)){
                connections.push_back(_tmp_conn);
            }
        }
        else //this should always happen for at least the leaf nodes because their variance is zero
        {
            // Determine if point is in a band by checking neighbor CPPN values
            if (outgoing)
            {
                left = Math.Abs(c.w - queryCPPN(a, b, c.x - node.width, c.y));
                right = Math.Abs(c.w - queryCPPN(a, b, c.x + node.width, c.y));
                top = Math.Abs(c.w - queryCPPN(a, b, c.x, c.y - node.width));
                bottom = Math.Abs(c.w - queryCPPN(a, b, c.x, c.y + node.width));
            }
            else
            {
                left = Math.Abs(c.w - queryCPPN(c.x - node.width, c.y, a, b));
                right = Math.Abs(c.w - queryCPPN(c.x + node.width, c.y, a, b));
                top = Math.Abs(c.w - queryCPPN(c.x, c.y - node.width, a, b));
                bottom = Math.Abs(c.w - queryCPPN(c.x, c.y + node.width, a, b));
            }

            if (std::max(std::min(top, bottom), std::min(left, right)) > bandThrehold)
            {
                if (outgoing)
                {
                    connections.emplace_back(a, b, c.x, c.y, c.w);
                }
                else
                {
                    connections.emplace_back(c.x, c.y, a, b, c.w);
                }
            }

        }
    }

    return connections;
}

void EvolvableSubstrate::generateSubstrate(int initialDepth, float varianceThreshold, float bandThreshold, int ESIterations,
                                        float divsionThreshold, int maxDepth,
                                        unsigned int inputCount, unsigned int outputCount)
{



    int sourceIndex = 0, targetIndex = 0;
    unsigned int counter = 0;

    this.initialDepth = initialDepth;
    this.maxDepth = maxDepth;
    this.varianceThreshold = varianceThreshold;
    this.bandThrehold = bandThreshold;
    this.divisionThreshold = divsionThreshold;


    std::vector<TempConnection> tempConnections;

    //CONNECTIONS DIRECTLY FROM INPUT NODES
    for (std::vector<PointD> & input: inputNeuronPositions)
    {
        // Analyze outgoing connectivity pattern from this input
        QuadPoint root = QuadTreeInitialisation(input.X, input.Y, true, (int)initialDepth, (int)maxDepth);

        // Traverse quadtree and add connections to list
        tempConnections = std::move(PruneAndExpress(input.X, input.Y, root, true, maxDepth));

        for (TempConnection & p : tempConnections)
        {
            PointD newp = PointD(p->x2, p->y2);

            targetIndex = hiddenCoordinates.IndexOf(newp);
            if (targetIndex == -1)
            {

                targetIndex = hiddenNeurons.Count;
                hiddenCoordinates.Add(newp);
            }
            connections.Add(new ConnectionGene(counter++, (uint)(sourceIndex), (uint)(targetIndex + inputCount + outputCount), p->weight * HyperNEATParameters.weightRange, new float[] {p->x1,p->y1,p->x2,p->y2}));
        }
        sourceIndex++;
    }

    tempConnections.Clear();

    //hidden to hidden part

    std::vector<PointD*> unexploredHiddenNodes;
    for(NEAT::PointD * p: hiddenNeurons)
        unexploredHiddenNodes.push_back(p);

    for (int step = 0; step < ESIterations; step++)
    {
        for (PointD *& hiddenP: unexploredHiddenNodes)
        {
            tempConnections.clear();
            QuadPoint root = QuadTreeInitialisation(hiddenp->X, hiddenp->Y, true, (int)initialDepth, (int)maxDepth);
            PruneAndExpress(hiddenp->first, hiddenp->Y, tempConnections, root, true, maxDepth);

            sourceIndex = hiddenNeurons.IndexOf(hiddenP);   //TODO there might a computationally less expensive way

            for (TempConnection p: tempConnections)
            {

                PointF newp = new PointF(p->x2, p->y2);

                targetIndex = hiddenCoordinates.IndexOf(newp);
                if (targetIndex == -1)
                {
                    targetIndex = hiddenNeurons.Count;
                    hiddenCoordinates.Add(newp);

                }
                m_links.emplace_back(counter++, (uint)(sourceIndex + inputCount + outputCount), (uint)(targetIndex + inputCount + outputCount), p->weight * HyperNEATParameters.weightRange, new float[] { p->x1, p->y1, p->x2, p->y2 });
            }
        }
        // Remove the just explored nodes
        std::vector<PointD> temp;
        temp->AddRange(hiddenNeurons);
        for (PointF f: unexploredHiddenNodes)
            temp->Remove(f);

        unexploredHiddenNodes = temp;

    }

    tempConnections.clear();

    //CONNECT TO OUTPUT
    targetIndex = 0;
    for (PointF & outputPos: outputNeuronPositions)
    {
        // Analyze incoming connectivity pattern to this output
        QuadPoint root = QuadTreeInitialisation(outputPos.X, outputPos.Y, false, (int)initialDepth, (int)maxDepth);
        tempConnections.Clear();
        PruneAndExpress(outputPos.X, outputPos.Y, tempConnections, root, false, maxDepth);

        for (TempConnection & t: tempConnections)
        {
            PointD source(t.x1, t.y1);
            sourceIndex = hiddenNeurons.IndexOf(source);

            /* New nodes not created here because all the hidden nodes that are
                connected to an input/hidden node are already expressed. */
            if (sourceIndex != -1)  //only connect if hidden neuron already exists
                connections.Add(new ConnectionGene(counter++, (uint)(sourceIndex + inputCount + outputCount), (uint)(targetIndex + inputCount), t.weight * HyperNEATParameters.weightRange, new float[] { t.x1, t.y1, t.x2, t.y2 }));
        }
        targetIndex++;
    }
}

float EvolvableSubstrate::queryCPPN(float x1, float y1, float x2, float y2)
{
    coordinates[0] = x1;
    coordinates[1] = y1;
    coordinates[2] = x2;
    coordinates[3] = y2;

    cppn.ClearSignals();
    cppn.SetInputSignals(coordinates);
    cppn.RecursiveActivation();

    return genome.GetOutputSignal(0);
}

void EvolvableSubstrate::getCPPNValues(std::vector<float> & l, QuadPoint & p)
{
    if (p && !p.childs.empty())
    {
        for (int i = 0; i < 4; i++)
        {
            getCPPNValues(l, p->childs[i]);
        }
    }
    else
    {
        l.push_back(p.w);
    }
}
