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

    public:
        using Covariance = typename ADEKF<State>::Covariance;
        using ScalarType = typename ADEKF<State>::ScalarType;
        static constexpr size_t DOF = DOFOf<State>;
        static constexpr double weighted_mean_epsilon = 1e-8;
        aligned_vector<State> old_mus, smoothed_mus, predicted_mus;
        aligned_vector<Covariance> old_sigmas, smoothed_sigmas, predicted_sigmas;

        RTS_Smoother(const State &_mu, const Covariance &_sigma) : ADEKF<State>(_mu, _sigma)
        {
            storeEstimation();
            storePredictedEstimation();
        }
        RTS_Smoother(const ADEKF<State> &filter) : ADEKF<State>(filter)
        {
            storeEstimation();
            storePredictedEstimation();
        }
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

        void storePredictedEstimation()
        {
            predicted_mus.push_back(this->mu);
            predicted_sigmas.push_back(this->sigma);
        }
        template <typename Head, typename... Tail>
        std::tuple<Tail...> tuple_tail(const std::tuple<Head, Tail...> &tuple)
        {
            return apply([](auto head, auto... tail) {
                return std::make_tuple(tail...);
            },
                         tuple);
        }

        template <typename DynamicModel, typename tstate, typename NoiseVector, typename... Controls, typename... ExpandedControls>
        void applyDynamicModel(DynamicModel dynamicModel, tstate &input, const NoiseVector &noise, const std::tuple<Controls...> &controls, const ExpandedControls &...expanded_controls)
        {
            if constexpr (std::tuple_size<std::tuple<Controls...>>::value > 0)
            {
                applyDynamicModel(dynamicModel, input, noise, tuple_tail(controls), expanded_controls..., std::get<0>(controls));
            }
            else
            {
                dynamicModel(input, noise, expanded_controls...);
            }
        }

    

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        Covariance getNonAdditiveDynamicNoise(const State &smoothed, DynamicModel dynamicModel, const Eigen::Matrix<double,NoiseDim,NoiseDim> &Q, const std::tuple<Controls...> &controls)
        {
            auto derivator = adekf::getDerivator<DOF + NoiseDim>();
            auto input = eval(smoothed + derivator.template head<DOF>());
            //Todo u vernünftig machen
            applyDynamicModel(dynamicModel, input, derivator.template tail<NoiseDim>(), controls);
            auto result = (input - smoothed);
            //std::cout << result << std::endl;
            auto F=extractJacobi(result);
            return eval(F.template rightCols<NoiseDim>() * Q * F.template rightCols<NoiseDim>().transpose());
        }
        //Taken from https://gist.github.com/gokhansolak/d2abaefcf3e3b767f5bc7d81cfe0b36a
        template <typename _Matrix_Type_>
        _Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
        {
            Eigen::JacobiSVD<_Matrix_Type_> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
            double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
            return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
        }

        

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothSingleStep(size_t k, DynamicModel &dynamicModel, const Eigen::Matrix<double,NoiseDim,NoiseDim> &Q, const std::tuple<Controls...> &controls, const State &smoothed_mu_kplus, const Covariance &smoothed_sigma_kplus)
        {
         
            auto delta = adekf::getDerivator<DOF>();
            auto input = eval(old_mus[k] + delta);
            applyDynamicModel(dynamicModel, input, MatrixType<ScalarType,NoiseDim, 1>::Zero(), controls);

            auto ref=old_mus[k]
            do{

            Covariance predicted_to_r=extractJacobi(predicted_mus[k+1]+delta - old_mus[k]);
            Covariance predicted_in_r=predicted_to_r*predicted_sigmas[k+1]*predicted_to_r.transpose();
            Covariance smoothedplus_to_r=extractJacobi(smoothed_mu_kplus+delta-old_mus[k]);
            Covariance smoothed_in_r=smoothedplus_to_r*smoothed_sigma_kplus*smoothedplus_to_r.transpose();

            Covariance Fk=extractJacobi(input -old_mus[k]);


            Covariance SmootherGain = old_sigmas[k] * Fk.transpose() * predicted_in_r.inverse();
            typename State::template DeltaType<double> smoother_innovation=SmootherGain * ((smoothed_mu_kplus-old_mus[k])-(predicted_mus[k + 1]-old_mus[k]));
            smoothed_mus[k] = old_mus[k] + smoother_innovation;

          
            auto result2 = eval(old_mus[k] + (smoother_innovation+delta) - smoothed_mus[k]);
             Covariance Jk=extractJacobi(result2);
           

            //Covariance Bk=extractJacobi(smoothed_mu_kplus+delta-predicted_mus[k+1]);
            /*smoothed_sigmas[k] = Jk * (old_sigmas[k] + SmootherGain * (smoothed_in_r-predicted_in_r) * SmootherGain.transpose())*Jk.transpose();
            assurePositiveDefinite(smoothed_sigmas[k]);*/
            Fk=extractJacobi(input-smoothed_mus[k]);
            Covariance old_to_r=extractJacobi(old_mus[k]+delta-smoothed_mus[k]);
            Covariance old_in_r=old_to_r*old_sigmas[k]*old_to_r.transpose();
            predicted_to_r=extractJacobi(predicted_mus[k+1]+delta - smoothed_mus[k]);
            predicted_in_r=predicted_to_r*predicted_sigmas[k+1]*predicted_to_r.transpose();
            smoothed_in_r=smoothed_sigma_kplus;
            SmootherGain = old_in_r * Fk.transpose() * predicted_in_r.inverse();
            smoothed_sigmas[k] = old_in_r + SmootherGain * (smoothed_in_r-predicted_in_r) * SmootherGain.transpose();
            assurePositiveDefinite(smoothed_sigmas[k]);
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothIntervalWithNonAdditiveNoise(size_t steps, int start, DynamicModel &dynamicModel, const Eigen::Matrix<double,NoiseDim,NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
        {
            if (start < 0)
            {
                start = old_mus.size() - 1 + start;
            }
            assert(start - steps >= 0 && "Can not smooth more estimates than available");
            assert(start < old_mus.size() - 1 && "Can not smooth a state which belongs to future");
            assert(all_controls.size() == old_mus.size() - 1 && "Requires all control inputs for the dynamic model.");
            for (size_t k = start; k >= old_mus.size() - steps - 1; k--)
            {
                //std::cout << "At iteration: " << k << std::endl;
                smoothSingleStep(k, dynamicModel, Q, all_controls[k], smoothed_mus[k + 1], smoothed_sigmas[k + 1]);
            }
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothAllWithNonAdditiveNoise(DynamicModel &dynamicModel, const Eigen::Matrix<double,NoiseDim,NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
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