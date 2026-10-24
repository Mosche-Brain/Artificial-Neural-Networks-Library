/**
 * @file ModelBase.hpp
 * @author jaro
 * @brief Base interface for high-level neural network models in YANN.
 * @date 2026-07-19
 */

#ifndef YANN_MODELBASE_HPP
#define YANN_MODELBASE_HPP

namespace yann::models
{
    /**
     * @class ModelBase
     * @brief Abstract base class representing a machine learning model architecture.
     */
    class ModelBase
    {
    public:
        virtual ~ModelBase() = default;
    };
} // yann::models

#endif //YANN_MODELBASE_HPP
