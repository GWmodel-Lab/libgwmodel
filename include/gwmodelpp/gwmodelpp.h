#ifndef PREFIX_H
#define PREFIX_H

#ifdef CREATECPP_EXPORTS
#ifdef WIN32
#ifdef CREATEDLL_EXPORTS
#define GWMODELPP_API __declspec(dllexport)
#else
#define GWMODELPP_API __declspec(dllimport)
#endif // CREATEDLL_EXPORTS
#else 
#define GWMODELPP_API  
#endif // WIN32
#else
#define GWMODELPP_API  
#endif // CREATECPP_EXPORTS

#endif  // PREFIX_H