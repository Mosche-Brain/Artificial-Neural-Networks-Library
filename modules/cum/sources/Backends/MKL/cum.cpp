#include "cum/cum.hpp"

#include "internal/context.hpp"

void cum::cum(DEVICE device)
{
    cum::internal::context(device);
}

void cum::recum(DEVICE device)
{
    cum::internal::context(device);
}

void cum::decum()
{

}