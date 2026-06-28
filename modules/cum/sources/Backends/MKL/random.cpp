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
        //
        // cum::library::getQueue().submit([&](sycl::handler& h) {
        //     h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        //         oneapi::mkl::rng::device::philox4x32x10<> engine(69, idx[0]);
        //         oneapi::mkl::rng::device::uniform<float> distr(min, max);
        //
        //         float x = oneapi::mkl::rng::device::generate(distr, engine);
        //         buff[idx] = static_cast<cumeric_t>(x);
        //     });
        // });

    }
}  