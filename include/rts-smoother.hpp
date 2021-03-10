#pragma once

//#include <ADEKF/ceres/jet.h>
#include "ADEKFUtils.h"
#include "ADEKF.h"

#include <iostream>
#include <Eigen/Core>
#include <vector>

namespace adekf
{

    //using namespace Eigen;
    //using namespace std::placeholders;

    /**
     * An RTS Smoother Implementation based on the ADEKF
     * @tparam Filter The Filter to be used for estimation
     */
    template <typename State>
    class RTS_Smoother : public ADEKF<State>
    {
        using Covariance = typename ADEKF<State>::Covariance;

    public:
        static constexpr size_t DOF = DOFOf<State>;
        static constexpr double weighted_mean_epsilon = 1e-8;
        std::vector<State, Eigen::aligned_allocator<State>> old_mus;
        std::vector<State, Eigen::aligned_allocator<State>> smoothed_mus;
        std::vector<Covariance, Eigen::aligned_allocator<Covariance>> old_sigmas;
        std::vector<Covariance, Eigen::aligned_allocator<Covariance>> smoothed_sigmas;

        RTS_Smoother(const State &_mu, const Covariance &_sigma) : ADEKF<State>(_mu, _sigma) { storeEstimation(); }
        RTS_Smoother(const ADEKF<State> &filter) : ADEKF<State>(filter) { storeEstimation(); }
        /**
         * Stores the current mean and covariance of the filter 
         */
        void storeEstimation()
        {
            old_mus.push_back(this->mu);
            smoothed_mus.push_back(this->mu);
            old_sigmas.push_back(this->sigma);
            smoothed_sigmas.push_back(this->sigma);
        }
        template <typename Head, typename... Tail>
        std::tuple<Tail...> tuple_tail(const std::tuple<Head, Tail...> &t)
        {
            return apply([](auto head, auto... tail) {
                return std::make_tuple(tail...);
            },
                         t);
        }

        template <typename DynamicModel, typename tstate, typename NoiseVector, typename... Controls, typename... ExpandedControls>
        void applyDynamicModel(DynamicModel dynamicModel, tstate &input, const NoiseVector &noise, const std::tuple<Controls...> &controls, const ExpandedControls& ...  expanded_controls)
        {
             if constexpr(std::tuple_size<std::tuple<Controls ...>>::value > 0)
            {
                applyDynamicModel(dynamicModel, input, noise, tuple_tail(controls), expanded_controls..., std::get<0>(controls));
            }
            else
            {
                dynamicModel(input, noise, expanded_controls...);
            }
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        Covariance getNonAdditiveDynamicNoise(const State &smoothed, DynamicModel dynamicModel, const SquareMatrixType<double, NoiseDim> &Q, const std::tuple<Controls...> &controls)
        {
            auto derivator = adekf::getDerivator<DOF + NoiseDim>();
            auto input = eval(smoothed + derivator.template head<DOF>());
            //Todo u vernünftig machen
            applyDynamicModel(dynamicModel, input, derivator.template tail<NoiseDim>(), controls);
            auto result = (input - smoothed);
            //std::cout << result << std::endl;
            Eigen::Matrix<double, DOF, DOF + NoiseDim> F(DOF, DOF + NoiseDim);
            //Eigen::Matrix<double,DOF,1> diff;
            for (size_t j = 0; j < DOF; ++j)
            {
                F.row(j) = result(j).v;
                //diff(j)=result(j).a;
            }
            return eval(F.template rightCols<NoiseDim>() * Q * F.template rightCols<NoiseDim>().transpose());
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothIntervalWithNonAdditiveNoise(size_t steps, int start, DynamicModel dynamicModel, const SquareMatrixType<double, NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
        {
            if (start < 0)
            {
                start = old_mus.size() - 1 + start;
            }
            assert(start - steps >= 0 && "Can not smooth more estimates than available");
            assert(start < old_mus.size() - 1 && "Can not smooth a state which belongs to future");
            assert(all_controls.size() == old_mus.size() - 1 && "Requires all control inputs for the dynamic model.");
            for (size_t i = start; i >= old_mus.size() - steps - 1; i--)
            {
                std::cout << "At iteration: " << i << std::endl;
                State smoothed = old_mus[i];
                Eigen::Matrix<double, DOF, DOF> J1(DOF, DOF), J2(DOF, DOF);
                Covariance DynamicCov, SmootherGain, PredictedCov;
                auto controls = all_controls[i];
                DynamicCov = getNonAdditiveDynamicNoise(smoothed, dynamicModel, Q, controls);

                auto derivator = adekf::getDerivator<DOF>();
                auto input = eval(smoothed + derivator);
                //Todo  u vernünftig machen
                applyDynamicModel(dynamicModel,input,Eigen::Matrix<double, NoiseDim, 1>::Zero(), controls);
                applyDynamicModel(dynamicModel,smoothed,Eigen::Matrix<double, NoiseDim, 1>::Zero(), controls);
                auto result1 = eval(input - smoothed);
                Eigen::Matrix<double, DOF, 1> diff1;

                for (size_t j = 0; j < DOF; ++j)
                {
                    J1.row(j) = result1(j).v;
                    diff1(j) = result1(j).a;
                }

                PredictedCov = J1 * old_sigmas[i] * J1.transpose() + DynamicCov;
                SmootherGain = old_sigmas[i] * J1.transpose() * PredictedCov.inverse();
                smoothed_mus[i] = old_mus[i] + (SmootherGain * (smoothed_mus[i + 1] - smoothed));
                auto result2 = eval(input - smoothed_mus[i]);

                for (size_t j = 0; j < DOF; ++j)
                {
                    J2.row(j) = result2(j).v;
                }

                smoothed_sigmas[i] = J2 * (old_sigmas[i] + SmootherGain * (smoothed_sigmas[i + 1] - PredictedCov) * SmootherGain.transpose()) * J2.transpose();
            }
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothAllWithNonAdditiveNoise(DynamicModel dynamicModel, const SquareMatrixType<double, NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
        {
            smoothIntervalWithNonAdditiveNoise(old_mus.size() - 2, -1, dynamicModel, Q, all_controls);
        }
    };

    /**
     * General Deduction Template for the RTS_Smoother based on StateRetriever.
     * This is needed so you can type RTS_Smoother smoother(Filter{...}) without template arguments
     */
    /*template <typename State>
    RTS_Smoother(ADEKF<State> &&) -> RTS_Smoother<State>;
    template <typename State>
    RTS_Smoother(ADEKF<State> ) -> RTS_Smoother<State>;
*/

} // namespace adekf