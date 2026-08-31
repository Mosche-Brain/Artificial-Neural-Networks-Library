#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "cum/Core.hpp"
#include "layers/Layers.hpp"
#include "logging/ITrainingCallback.hpp"
#include "loss/LossBase.hpp"
#include "optimizers//OptimizerBase.hpp"

/* # Used conventions
 * - Layer indexing starts from 0
 * - Vectors are 1D matrices
 * - Inputs and results are column vectors
 * - Training data are transposed (every row is a sample, it is transposed in training)
 */

namespace yann::models
{
	struct Sample
	{
		cum::Matrix X;
		cum::Matrix Y;
	};

	struct SampleRef
	{
		cum::Matrix& X;
		cum::Matrix& Y;
	};

	struct Batch
	{
		enum class ORIENTATION { ROW_SAMPLE, COLUMN_SAMPLE };
		ORIENTATION orientation = ORIENTATION::COLUMN_SAMPLE;

		cum::Matrix X;
		cum::Matrix Y;
		cum::dim_t size;

		const cum::Matrix& inputs() const noexcept
		{
			return X;
		}

		const cum::Matrix& targets() const noexcept
		{
			return Y;
		}

        Sample sample(size_t index) const
        {
            return orientation == ORIENTATION::ROW_SAMPLE ? Sample{ X.row(index), Y.row(index) } : Sample{ X.col(index), Y.col(index) };
        }

        cum::Matrix x(size_t index) const
        {
            return orientation == ORIENTATION::ROW_SAMPLE ? X.row(index) : X.col(index);
        }

        cum::Matrix y(size_t index) const
        {
            return orientation == ORIENTATION::ROW_SAMPLE ? Y.row(index) : Y.col(index);
        }

        Batch(cum::Matrix X, cum::Matrix Y, ORIENTATION orientation = ORIENTATION::COLUMN_SAMPLE)
            : orientation(orientation),
              X(std::move(X)),
              Y(std::move(Y)),
              size(orientation == ORIENTATION::ROW_SAMPLE ? this->X.rows() : this->X.cols())
        {
        }
	};

    class Sequential
    {
    public:
        using Topology = std::vector<std::unique_ptr<layers::LayerBase>>;
        using LayerPtr = std::unique_ptr<layers::LayerBase>;

        /* Constructors */
        Sequential() ;
        Sequential(std::initializer_list<LayerPtr> newTopology, bool build=true);

        /* Methods */
        
        cum::Matrix forward(const cum::Matrix& input);
        
        void fit(const cum::Matrix& X, const cum::Matrix& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, size_t epochs, size_t batch_size=1, std::span<logging::ITrainingCallback*> callbacks = {});
        void addLayer(LayerPtr layer);
        void clear();

        void build();
        /* Getters */
        cum::Matrix& getWeights(size_t layer) const;
        cum::Matrix& getBiases(size_t layer) const;
        cum::Matrix& getOutputs(size_t layer) const;
        cum::functions::activation_t getActivation(size_t layer) const;
        auto getTopology() const -> const Topology&;
        auto getLayer(size_t layer) const -> LayerPtr;
        size_t getLayersCount() const;
        int getLayerSize(size_t layer) const;

        std::vector<Parameter*> parameters();
    protected:
        void backward(const cum::Matrix& d_output);

        Topology topology;
    };
}
