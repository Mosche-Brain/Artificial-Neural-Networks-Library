//
// Created by jaro on 7/31/26.
//

#ifndef YANN_LOSSBASE_HPP
#define YANN_LOSSBASE_HPP

#include <memory>
#include <stdexcept>

#include "cum/Core.hpp"
#include "cum/Matrix.hpp"
#include "cum/Tensor.hpp"

namespace yann::loss
{
    typedef struct
    {
        cum::cumeric_t value;
        cum::Tensor gradient;
    } loss_t;


    //! Sposób redukcji straty i jej gradientu po batchu.
    //! W obu wariantach strata jest sumowana po cechach.
    enum class Reduction
    {
        BatchMean, //!< suma po cechach, średnia po batchu (domyślnie)
        Sum        //!< suma po cechach i po batchu
    };

    class LossBase
    {
    public:
        //! Oś batcha jest stała dla całego frameworka: tensory mają układ [batch, ...].
        static constexpr cum::dim_t batch_axis = 0;

        virtual ~LossBase() = default;

        //! Waliduje kształty, wywołuje compute_sum i stosuje redukcję.
        //! Metoda jest niewirtualna, więc domyślny argument jest bezpieczny.
        void compute(const cum::Tensor& predicted, const cum::Tensor& target, Reduction reduction = Reduction::BatchMean);

        const loss_t& result() const { return loss; }
        cum::cumeric_t getLoss() const { return loss.value; }
        cum::Tensor getGradient() const { return loss.gradient; }
    protected:
        //! Liczy stratę i gradient jako sumy po cechach i po batchu, bez skalowania.
        virtual void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) = 0;

        loss_t loss;
    };

    using Loss = std::unique_ptr<LossBase>;

} // yann

#endif //YANN_LOSSBASE_HPP
