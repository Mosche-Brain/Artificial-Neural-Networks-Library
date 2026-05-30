#include "cum/random.hpp"
#include "cumMKL.hpp"

#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/rng/device.hpp>

namespace cum
{
    void random::uniform(cumeric_t* buff, size_t N, cumeric_t min, cumeric_t max)
    {
        oneapi::mkl::rng::device::philox4x32x10 engine;
        oneapi::mkl::rng::device::uniform dist((float)min, (float)max);

        for(size_t i = 0 ; i < N ; i++)
        {
            buff[i] = (cumeric_t)oneapi::mkl::rng::device::generate(dist, engine);
        }

    }
}