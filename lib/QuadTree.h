#ifndef QUADTREE_H
#define QUADTREE_H

#include <tr1/memory>
#include <vector>
#include <stack>



template <typename T>
struct SQuadPoint
{
    struct QuadIterator: public std::iterator<std::input_iterator_tag, SQuadPoint>
    {
      SQuadPoint * current;
      std::stack<SQuadPoint*> stack;

    public:
      QuadIterator(SQuadPoint * x=nullptr) :current(x) {
         stack.push(current);
         if(current != nullptr && !(current->isLeaf())){
             (*this)++;
         }
      }
      QuadIterator(const SQuadPoint * mit) : current(mit->current), stack(mit->stack){}
      bool operator==(const QuadIterator& rhs) {return current == rhs.current;}
      bool operator!=(const QuadIterator& rhs) {return current != rhs.current;}
      T & operator*() {return  current->points[0];}
      void operator++(int){
          SQuadPoint * before = current;
          while (!stack.empty() && (current->isLeaf() ? (current == before): !(current->isLeaf()))) {
             current = stack.top();
             stack.pop();

             //Visit the node pointed to by traverse.

             /*
              * If there is a left child, add it
              * for later processing.
              */
             for(ushort i=0; i<=TOPRIGHT; ++i){
                 if (current->childs[i] != nullptr)
                   stack.push(current->childs[i].get());
             }
        }
          if(stack.empty()){
              current = nullptr;
          }
      }

    };

    enum position { TOPLEFT, BOTTOMLEFT, BOTTOMRIGHT, TOPRIGHT};
    float x, y;
    float width; //width of this quadtree square
    std::vector<std::shared_ptr<SQuadPoint>> childs;
    uint level; //the level in the quadtree
    SQuadPoint * parent;
    std::vector<T> points;

    SQuadPoint(float _x, float _y, float _w, int _level, SQuadPoint * _parent=nullptr):
        parent(_parent)
    {
        level = _level;
        points.reserve(1);
        x = _x;
        y = _y;
        width = _w;
        childs.resize(4, nullptr);
    }

    bool hasData(){
        return points.size() > 0;
    }

    bool isLeaf(){
        bool result = true;
        for(ushort i=0;i<childs.size();i++){
            if (childs[i] != nullptr){
                result = false;
                break;
            }
        }
        return result && (level != 0);
    }

    position getPosIndex(typename T::value_type _x, typename T::value_type _y){
        position pos;
        //calculate where point should be
        if(_x < x   && _y >= y ){// top left
            pos = TOPLEFT;
        }
        else if(_x <= x  &&  _y < y){
            pos = BOTTOMLEFT;
        }
        else if(_x > x && _y <= y){
            pos = BOTTOMRIGHT;
        }
        else if(_x >= x && _y > y){
            pos = TOPRIGHT;
        }
        else if(_x == x && _y == y){
            pos = TOPLEFT;
        }
        else{
            throw std::runtime_error("unexpected result inserting point");
        }
        return pos;
    }

    void initSQuadPoint(position pos){
        if(childs[pos] != nullptr){
            return;
        }
        switch(pos){
        case TOPLEFT:
            // Divide into sub-regions and assign children to parent
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x - width / 2, y + width / 2, width / 2, level + 1, this));
            break;
        case BOTTOMLEFT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x - width / 2, y - width / 2, width / 2, level + 1, this));
            break;
        case BOTTOMRIGHT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x + width / 2, y - width / 2, width / 2, level + 1, this));
            break;
        case TOPRIGHT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x + width / 2, y + width / 2, width / 2, level + 1, this));
            break;
        default:
            throw std::runtime_error("unexpected position inserting new SQuadPoint");
        }
    }

    bool _insertCreateQuad(T a_point){
        position pos = getPosIndex(a_point.X, a_point.Y);
        initSQuadPoint(pos);
        return childs[pos]->insert(a_point);
    }

    bool insert(T & a_point){
        bool result = false;
        if(std::fabs(a_point.X)  <= std::fabs(x) + width && std::fabs(a_point.Y)  <= std::fabs(y) + width){
            position pos = getPosIndex(a_point.X, a_point.Y);
            if(!isLeaf()){
                initSQuadPoint(pos);
                result = childs[pos]->insert(a_point);
            }
            else {
                if(hasData() && (points[0] != a_point)){
                    result = _insertCreateQuad(std::move(points[0]));
                    points.pop_back();
                    result = result && _insertCreateQuad(a_point);
                }
                else{
                    points.push_back(a_point);
                    result = true;
                }

            }
        }
        return result;
    }

    bool erase(QuadIterator & it){
        SQuadPoint & current = *it;
        SQuadPoint * parent = current.parent;
        if (current.isLeaf() && current.level != 0){
            SQuadPoint * parent = current.parent;
            short index = -1;
            for(ushort i=0; i<parent->childs.size();i++){
                if (parent->childs[i] == *it){
                    index = i;
                }
            }
            parent->childs.erase(index);
            if(parent->isLeaf() && !(parent->hasData())){
               return erase(QuadIterator(parent));
           }
           return true;
        }
        return false;
    }


    std::pair<QuadIterator, QuadIterator> find(T & a_point){
        std::pair<QuadIterator, QuadIterator> result;
        if(isLeaf()){
            if(hasData() && points[0] == a_point){
                result =  std::pair<QuadIterator, QuadIterator>(QuadIterator(this), QuadIterator());
            }
        }
        else{
            position i = getPosIndex(a_point.X, a_point.Y);
            if ( childs[i] != nullptr ){
                result = childs[i]->find(a_point);
            }
        }
        return result;
    }

    QuadIterator begin(){
        return QuadIterator(this);
    }

    QuadIterator end(){
        return QuadIterator();
    }
};



#endif // QUADTREE_H
