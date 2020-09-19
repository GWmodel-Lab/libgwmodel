#ifndef GWMODEL_H
#define GWMODEL_H

#ifdef CREATE_SHARED_LIB
#ifdef WIN32
#ifdef CREATE_DLL_EXPORTS
#define GWMODEL_API __declspec(dllexport)
#else
#define GWMODEL_API __declspec(dllimport)
#endif 
#else
#define GWMODEL_API  
#endif
#else
#define GWMODEL_API  
#endif

class CGwmSpatialAlgorithm;
    
extern "C" GWMODEL_API CGwmSpatialAlgorithm* gwmodel_create_algorithm();

#endif  // GWMODEL_H