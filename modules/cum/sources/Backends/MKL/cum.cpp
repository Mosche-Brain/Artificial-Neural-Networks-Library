#include "cum/cum.hpp"

#include "cumMKL.hpp"

void cum::cum(CUM_DEVICE device)
{
    cum::library::getContext().setDevice(device);
}

void cum::recum(CUM_DEVICE device)
{
    cum::library::getContext().setDevice(device);
}

void cum::decum()
{

}