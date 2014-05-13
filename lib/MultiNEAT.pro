SOURCES += *.cpp \
    Point.cpp

HEADERS += *.h \
    SubstrateBase.h \
    Const.h \
    QuadTree.h
TEMPLATE = lib
CONFIG += c++11
CONFIG -= qt
CONFIG += dll
TARGET = _MultiNEAT
INCLUDEPATH += /usr/include/python2.7
QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -Wextra
QMAKE_CXXFLAGS += -Wshadow
QMAKE_CXXFLAGS += -Wuninitialized
QMAKE_CXXFLAGS += -Wno-unused

QMAKE_CXX = ccache g++

unix:!macx: LIBS += -lboost_serialization

unix:!macx: LIBS += -lboost_python
