#include "cum/cum.hpp"


void cum::cum(CUM_DEVICE device)
{
    cum::internal::getContext().setDevice(device);
}

void cum::recum(CUM_DEVICE device)
{
    cum::internal::getContext().setDevice(device);
}

void cum::decum()
{

}