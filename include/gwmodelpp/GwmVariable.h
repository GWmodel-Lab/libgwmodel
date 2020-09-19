#ifndef GWMVARIABLE_H
#define GWMVARIABLE_H

#include <string>

struct GwmVariable
{
    int index;
    bool isNumeric;
    std::string name;

    GwmVariable() : index(-1), isNumeric(false), name() {}
    GwmVariable(int i, bool numeric, std::string n) : index(i), isNumeric(numeric), name(n) {}
    GwmVariable(const GwmVariable& v) : index(v.index), isNumeric(v.isNumeric), name(v.name) {}

    bool operator==(const GwmVariable& right) const
    {
        return index == right.index && isNumeric == right.isNumeric && name == right.name;
    }

    bool operator!=(const GwmVariable& right) const
    {
        return index != right.index || isNumeric != right.isNumeric || name == right.name;
    }
};

#endif  // GWMVARIABLE_H