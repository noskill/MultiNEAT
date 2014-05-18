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
      QuadIterator(SQuadPoint * x=nullptr) :current(x){
         if (current != nullptr){
            stack.push(current);
         }
         if(current != nullptr && !(current->isLeaf())){
             (*this)++;
         }
      }
      QuadIterator(const SQuadPoint * mit) :
          current(mit->current), stack(mit->stack)
      {}
      bool operator==(const QuadIterator& rhs) {return current == rhs.current;}
      bool operator!=(const QuadIterator& rhs) {return current != rhs.current;}
      T & operator*() {return  current->points[0];}
      void operator++(int){


          SQuadPoint * before = current;
          // if current == before and current is leaf search for new leaf

          while (!stack.empty() && ((current == before) || !current->isLeaf())) {
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

          if(stack.empty() && current == before){
              current = nullptr;
              return;

        }

      }
      void operator++(){
          (*this)++;
      }
      T * operator->(){
          return &(current->points[0]);
      }
    };

    enum position { TOPLEFT, BOTTOMLEFT, BOTTOMRIGHT, TOPRIGHT};
    float x, y;
    float width; //width of this quadtree square
    std::vector<std::shared_ptr<SQuadPoint>> childs;
    uint level; //the level in the quadtree
    std::vector<T> points;
    size_t _size;

    SQuadPoint(float _x, float _y, float _w, int _level):
        _size(0)
    {
        level = _level;
        points.reserve(1);
        x = _x;
        y = _y;
        width = _w;
        childs.resize(4, nullptr);
    }

    SQuadPoint (const SQuadPoint & rhs):
    x(rhs.x),
    y(rhs.y),
    width(rhs.width),
    level(rhs.level),
    points(rhs.points),
    _size(rhs._size)
    {
        points.reserve(1);
        childs.resize(4, nullptr);
        for(size_t i=0; i<childs.size(); i++){
            if(rhs.childs[i] != nullptr){
                childs[i] = std::make_shared<SQuadPoint<T>>(*(rhs.childs[i]));
                assert(childs[i] != nullptr);
            }
        }
    }

//    SQuadPoint operator=(const SQuadPoint && rhs)=delete;

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
        return result;
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
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x - width / 2, y + width / 2, width / 2, level + 1));
            break;
        case BOTTOMLEFT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x - width / 2, y - width / 2, width / 2, level + 1));
            break;
        case BOTTOMRIGHT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x + width / 2, y - width / 2, width / 2, level + 1));
            break;
        case TOPRIGHT:
            childs[pos] = std::unique_ptr<SQuadPoint>(new SQuadPoint(x + width / 2, y + width / 2, width / 2, level + 1));
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

    typename T::data_type & operator[](const T & a_point){
        auto it = find(a_point);
        if(it == end()){
            this->insert(a_point);
        }
        it = find(a_point);
        return (*it).data;
    }

    bool insert(const T & a_point){
        bool result = false;
        if(std::fabs(a_point.X)  <= std::fabs(x) + width && std::fabs(a_point.Y)  <= std::fabs(y) + width){
            position pos = getPosIndex(a_point.X, a_point.Y);
            if(!isLeaf()){
                initSQuadPoint(pos);
                result = childs[pos]->insert(a_point);
            }
            else {
                if(hasData()){
                    if((points[0] != a_point)){
                        result = _insertCreateQuad(std::move(points[0]));
                        points.pop_back();
                        result = result && _insertCreateQuad(a_point);
                    }
                }
                else{
                    points.push_back(a_point);
                    result = true;
                }

            }
        }
        if(result){
            _size++;
        }
        return result;
    }

    bool erase (T & a_point){
        return erase(find(a_point));
    }

    bool erase(QuadIterator it){
        SQuadPoint * current = it.stack.top();it.stack.pop();
        if(current->isLeaf() && it.current == current){
            current->points.clear();
            current->_size = 0;
            return true;
        }
        SQuadPoint * parent = it.stack.top();
        short index = -1;
        for(ushort i=0; i<parent->childs.size();i++){
            if (parent->childs[i].get() == current){
                index = i;
                break;
            }
        }
        parent->childs[index].reset();

        while(it.stack.size() >= 2 && parent->isLeaf() && !parent->hasData()){
            current = it.stack.top();it.stack.pop();
            parent = it.stack.top();
            index = -1;
            for(ushort i=0; i<parent->childs.size();i++){
                if (parent->childs[i].get() == current){
                    index = i;
                    break;
                }
            }
            parent->childs[index].reset();
        }
        _size--;
        return true;
    }

    void clear(){
        for(size_t i=0; i < childs.size(); i++){
            childs[i].reset();
        }
        _size = 0;
        this->points.clear();

    }

    QuadIterator find(const T & a_point){
        assert(level == 0 || !isLeaf());
        QuadIterator result;
        SQuadPoint * tmp = this;

        while(true){
            result.stack.push(tmp);
            position i = tmp->getPosIndex(a_point.X, a_point.Y);
            if (tmp->childs[i] != nullptr ){
                tmp = tmp->childs[i].get();
            }
            else{
                break;
            }
        };

        if(tmp->hasData() && tmp->points[0] == a_point){
            result.current=tmp;
        }else{
            while(result.stack.size()){
                result.stack.pop();
            }
        }

        return result;
    }

    QuadIterator begin(){
        if(isLeaf() && !hasData()){
            return QuadIterator();
        }
        return QuadIterator(this);
    }

    QuadIterator end(){
        return QuadIterator();
    }

    size_t size()const{
        return _size;
    }

    bool checkSize(){
        uint vda=0;
        for(auto it = this->begin(); it != this->end();it++){
            vda++;
        }
        assert(vda == size());
        return vda == size();
    }
};



#endif // QUADTREE_H
