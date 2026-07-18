#include "ModelSession.hpp"

#include <algorithm>
#include <utility>

ModelSession::ModelSession(yann::models::Sequential model, yann::optimizers::Optimizer optimizer)
    : model_(std::move(model)), optimizer_(std::move(optimizer))
{

}

ModelSession::~ModelSession()
{
    if (trainThread_.joinable())
        trainThread_.join();
}

void ModelSession::setTrainingData(cum::Matrix X, cum::Matrix Y)
{
    std::lock_guard<std::mutex> lock(mutex_);
    X_train_ = std::move(X);
    Y_train_ = std::move(Y);
    cacheTrainTargetsLocked();
    curveDirty_ = true;
}

void ModelSession::setEvalInputs(cum::Matrix X)
{
    std::lock_guard<std::mutex> lock(mutex_);
    X_eval_ = std::move(X);
    curveDirty_ = true;
}

void ModelSession::setLossFunction(yann::utils::loss::LossFunction loss)
{
    std::lock_guard<std::mutex> lock(mutex_);
    lossFunction_ = loss;
    model_.setLossFunction(loss);
}


void ModelSession::setLearningRate(cum::cumeric_t rate)
{
    std::lock_guard<std::mutex> lock(mutex_);
    optimizer_->learning_rate = rate;
}

void ModelSession::reset(yann::models::Sequential model)
{
    if (isTraining_.load())
        return;

    if (trainThread_.joinable())
        trainThread_.join();

    std::lock_guard<std::mutex> lock(mutex_);
    model_ = std::move(model);
    model_.setLossFunction(lossFunction_);
    lastSnap_ = {};
    curveDirty_ = true;
}

float ModelSession::toF(cum::cumeric_t v)
{
    return static_cast<float>(v);
}

void ModelSession::cacheTrainTargetsLocked()
{
    curve_.trainXs.resize(X_train_.rows());
    curve_.trainYs.resize(Y_train_.rows());
    const std::size_t n = std::min(X_train_.rows(), Y_train_.rows());
    for (std::size_t i = 0; i < n; ++i)
    {
        curve_.trainXs[i] = toF(X_train_(i, 0));
        curve_.trainYs[i] = toF(Y_train_(i, 0));
    }
    curve_.trainXs.resize(n);
    curve_.trainYs.resize(n);
}

void ModelSession::recomputeCurveLocked()
{
    const std::size_t n = X_eval_.rows();
    curve_.xs.resize(n);
    curve_.ys.resize(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const cum::cumeric_t xin = X_eval_(i, 0);
        cum::Matrix x(1, 1, {xin});
        const cum::Matrix y = model_.forward(x);
        const cum::Matrix& inputActs = model_.getOutputs(0);
        curve_.xs[i] = inputActs.rows() > 0 ? toF(inputActs(0, 0)) : toF(xin);
        curve_.ys[i] = y.rows() > 0 ? toF(y(0, 0)) : 0.0f;
    }

    const std::size_t nTrain = std::min(X_train_.rows(), Y_train_.rows());
    curve_.trainPredYs.resize(nTrain);
    double sse = 0.0;
    for (std::size_t i = 0; i < nTrain; ++i)
    {
        const cum::cumeric_t xin = X_train_(i, 0);
        cum::Matrix x(1, 1, {xin});
        const cum::Matrix y = model_.forward(x);
        const float pred = y.rows() > 0 ? toF(y(0, 0)) : 0.0f;
        curve_.trainPredYs[i] = pred;
        const float target = toF(Y_train_(i, 0));
        const double d = static_cast<double>(pred - target);
        sse += d * d;
    }
    curve_.mse = nTrain > 0 ? static_cast<float>(sse / static_cast<double>(nTrain)) : 0.0f;

    curve_.valid = n > 0 || nTrain > 0;
    curveDirty_ = false;
}

NetworkSnapshot ModelSession::capture(yann::models::Sequential& model)
{
    NetworkSnapshot snap;
    const std::size_t layers = model.getLayersCount();
    snap.layerSizes.resize(layers);
    snap.weights.resize(layers);
    snap.outputs.resize(layers);

    for (std::size_t i = 0; i < layers; ++i)
    {
        snap.layerSizes[i] = std::max(1, model.getLayerSize(i));

        const cum::Matrix& outs = model.getOutputs(i);
        snap.outputs[i].resize(static_cast<std::size_t>(snap.layerSizes[i]), 0.0f);
        for (int n = 0; n < snap.layerSizes[i] && n < static_cast<int>(outs.rows()); ++n)
            snap.outputs[i][static_cast<std::size_t>(n)] = toF(outs(static_cast<std::size_t>(n), 0));

        if (i == 0)
            continue;

        const cum::Matrix& W = model.getWeights(i);
        snap.weights[i].assign(W.rows(), std::vector<float>(W.cols(), 0.0f));
        for (std::size_t r = 0; r < W.rows(); ++r)
            for (std::size_t c = 0; c < W.cols(); ++c)
                snap.weights[i][r][c] = toF(W(r, c));
    }

    if (layers > 0 && !snap.outputs.back().empty())
        snap.yHat = snap.outputs.back()[0];
    snap.valid = true;
    return snap;
}

NetworkSnapshot ModelSession::updateSnapshot(std::size_t sampleIndex)
{
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || X_eval_.rows() == 0)
        return lastSnap_;

    if (sampleIndex >= X_eval_.rows())
        sampleIndex = 0;

    if (curveDirty_)
        recomputeCurveLocked();

    const cum::cumeric_t xin = X_eval_(sampleIndex, 0);
    cum::Matrix x(1, 1, {xin});
    model_.forward(x);
    lastSnap_ = capture(model_);
    lastSnap_.x = lastSnap_.outputs.empty() || lastSnap_.outputs[0].empty()
        ? toF(xin)
        : lastSnap_.outputs[0][0];
    lastSnap_.yHat = lastSnap_.outputs.empty() || lastSnap_.outputs.back().empty()
        ? 0.0f
        : lastSnap_.outputs.back()[0];

    curve_.markerX = lastSnap_.x;
    curve_.markerY = lastSnap_.yHat;
    return lastSnap_;
}

void ModelSession::startTraining(cum::cumeric_t rate, std::size_t epochs)
{
    if (isTraining_.load())
        return;

    if (trainThread_.joinable())
        trainThread_.join();

    isTraining_.store(true);
    trainThread_ = std::thread([this, epochs]()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model_.fit(X_train_, Y_train_, *optimizer_, epochs);
        curveDirty_ = true;
        isTraining_.store(false);
    });
}

float ModelSession::evalInput(std::size_t sampleIndex) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (X_eval_.rows() == 0)
        return 0.0f;
    if (sampleIndex >= X_eval_.rows())
        sampleIndex = 0;
    return toF(X_eval_(sampleIndex, 0));
}

std::size_t ModelSession::evalSampleCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return X_eval_.rows();
}
