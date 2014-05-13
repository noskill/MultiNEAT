#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "vector"
#include "Assert.h"
#include <cmath>
#include <stdexcept>

#define point_to_vec(vec, point) \
    vec.push_back(point->X);\
    vec.push_back(point->Y);

namespace NEAT{

template<typename T, typename D>
struct Point2D
{
    T X;
    T Y;
    D data;

    static const unsigned short SIZE = 2;
    constexpr static const float THRESHOLD = 0.000001;
    typedef T value_type;

    Point2D(T x, T y):
        X(x), Y(y){}

    Point2D(std::vector<T> vec){
        ASSERT(vec.size() == 2);
        X = vec[0];Y = vec[1];
    }

    std::vector<T> vector(){
        std::vector<T> result;
        result.reserve(2);
        result.push_back(X);
        result.push_back(Y);
        return result;
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

    bool operator==(const Point2D & rhs){
        return (rhs.X == this->X) && (rhs.Y == this->Y);
    }

    bool operator!=(const Point2D & rhs){
        return !(this->operator ==(rhs));
    }

    unsigned short size() const {
        return SIZE;
    }

};

typedef Point2D<float, unsigned int> PointF;
typedef Point2D<double, unsigned int> PointD;

}


#endif // DEFINITIONS_H
