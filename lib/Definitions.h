#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "vector"

namespace NEAT{

template<typename T>
struct Point{

    T X;
    T Y;
    Point(T x, T y):
        X(x), T(y){}
    Point(std::vector<T> vec){
        ASSERT(vec.size() == 2);
        X = vec[0];Yv = vec[1];
    }
};

typedef Point<float> PointF;
typedef Point<double> PointD;

}


#endif // DEFINITIONS_H
