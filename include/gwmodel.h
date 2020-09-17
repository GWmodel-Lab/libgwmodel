#ifndef GWMODEL_H
#define GWMODEL_H

#ifdef WIN32
#ifdef CREATEDLL_EXPORTS
#define GWMODEL_API __declspec(dllexport)
#else
#define GWMODEL_API __declspec(dllimport)
#endif 
#else
#define GWMODEL_API  
#endif

class CGwmSpatialAlgorithm;
    
extern "C" GWMODEL_API CGwmSpatialAlgorithm* gwmodel_create_algorithm();

#endif  // GWMODEL_H