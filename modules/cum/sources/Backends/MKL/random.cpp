#include "cum/random.hpp"
#include "internal/context.hpp"

#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/rng/device.hpp>

namespace cum
{
    void random::uniform(cumeric_t* buff, size_t N, cumeric_t min, cumeric_t max, dim_t seed)
    {
        // oneapi::mkl::rng::device::philox4x32x10 engine(internal::getQueue(), seed);
        // oneapi::mkl::rng::device::philox4x32x10 engine;
        oneapi::mkl::rng::device::philox4x32x10<1>& engine = internal::getRNG();
        oneapi::mkl::rng::device::uniform dist((float)min, (float)max);

        for(size_t i = 0 ; i < N ; i++)
        {
            buff[i] = (cumeric_t)(oneapi::mkl::rng::device::generate(dist, engine));
        }
        //
        // cum::internal::getQueue().submit([&](sycl::handler& h) {
        //     h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        //         oneapi::mkl::rng::device::philox4x32x10<> engine(69, idx[0]);
        //         oneapi::mkl::rng::device::uniform<float> distr(min, max);
        //
        //         float x = oneapi::mkl::rng::device::generate(distr, engine);
        //         buff[idx] = static_cast<cumeric_t>(x);
        //     });
        // });

    }

    void random::normal(cumeric_t* buff, size_t N, cumeric_t sigma, std::size_t seed)
    {
        // oneapi::mkl::rng::device::philox4x32x10 engine(internal::getQueue(), 777);
        // oneapi::mkl::rng::device::gaussian<> dist((float)0, (float)sigma);

        constexpr int VecSize = 4;
        internal::getQueue().parallel_for(sycl::range<1>(N + VecSize - 1), [=](sycl::item<1> item) {

            // constexpr int VecSize = 1;

            oneapi::mkl::rng::device::philox4x32x10<VecSize>
                engine(seed, item.get_linear_id() * VecSize);

            oneapi::mkl::rng::device::gaussian<float> dist(0.0f, sigma);
            auto rnd = oneapi::mkl::rng::device::generate(dist, engine);

            size_t base = item.get_linear_id() * VecSize;

            #pragma unroll
            for (int i = 0; i < VecSize && base + i < N; ++i)
                buff[base + i] = rnd[i];
        }).wait();
    }
}  