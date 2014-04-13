#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "vector"
#include "Assert.h"
#include "math.h"
#include <stdexcept>

#define point_to_vec(vec, point) \
    vec.push_back(point->X);\
    vec.push_back(point->Y);

namespace NEAT{

template<typename T>
struct Point{
    T X;
    T Y;
    Point(T x, T y):
        X(x), Y(y){}
    Point(std::vector<T> vec){
        ASSERT(vec.size() == 2);
        X = vec[0];Y = vec[1];
    }

    std::vector<T> vector(){
        std::vector<T> result;
        result.push_back(X);
        result.push_back(Y);
        return result;
    }

    T distance() const {
        return X + Y;
    }

    bool operator==(const Point & other)const{
        return distance() == other.distance();
    }

    bool operator <(const Point & other)const{
        return distance() < other.distance();
    }

    bool operator >(const Point & other)const{
        return distance() > other.distance();
    }

    T operator[](uint index){
        switch(index){
        case 0:
            return X;
        case 1:
            return Y;
        default:
            throw std::out_of_range("index is out of range");
        }
    }

};

typedef Point<float> PointF;
typedef Point<double> PointD;

}


#endif // DEFINITIONS_H
