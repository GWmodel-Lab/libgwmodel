#ifndef CGWMALGORITHM_H
#define CGWMALGORITHM_H

class CGwmAlgorithm
{
public:
    CGwmAlgorithm();
    ~CGwmAlgorithm();

public:
    virtual void run() = 0;
};

#endif  // CGWMALGORITHM_H