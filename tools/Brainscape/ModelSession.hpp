#pragma once

#include "NetworkSnapshot.hpp"
#include "ResponseCurve.hpp"

#include <YANN/Models/Sequential.hpp>

#include <atomic>
#include <mutex>
#include <thread>

class ModelSession
{
public:
    explicit ModelSession(yann::models::Sequential model);
    ~ModelSession();

    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    void setTrainingData(cum::Matrix X, cum::Matrix Y);
    void setEvalInputs(cum::Matrix X);
    void setLossFunction(yann::utils::loss::LossFunction loss);
    yann::utils::loss::LossFunction lossFunction() const { return lossFunction_; }

    void reset(yann::models::Sequential model);

    NetworkSnapshot updateSnapshot(std::size_t sampleIndex);
    const NetworkSnapshot& lastSnapshot() const { return lastSnap_; }
    const ResponseCurve& responseCurve() const { return curve_; }

    void startTraining(cum::cumeric_t rate, std::size_t epochs);
    bool isTraining() const { return isTraining_.load(); }

    float evalInput(std::size_t sampleIndex) const;
    std::size_t evalSampleCount() const;

private:
    static float toF(cum::cumeric_t v);
    static NetworkSnapshot capture(yann::models::Sequential& model);
    void recomputeCurveLocked();
    void cacheTrainTargetsLocked();

    yann::models::Sequential model_;
    cum::Matrix X_train_;
    cum::Matrix Y_train_;
    cum::Matrix X_eval_;
    yann::utils::loss::LossFunction lossFunction_ = yann::utils::loss::LossFunction::mse;

    mutable std::mutex mutex_;
    std::atomic<bool> isTraining_{false};
    std::thread trainThread_;
    NetworkSnapshot lastSnap_;
    ResponseCurve curve_;
    bool curveDirty_ = true;
};
