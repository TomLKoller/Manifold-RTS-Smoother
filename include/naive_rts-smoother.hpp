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
    class Naive_RTS_Smoother : public ADEKF<State>
    {
       

    public:
        using Covariance = typename ADEKF<State>::Covariance;
        static constexpr size_t DOF = DOFOf<State>;
        static constexpr double weighted_mean_epsilon = 1e-8;
        aligned_vector<State> old_mus,smoothed_mus, predicted_mus;
        aligned_vector<Covariance> old_sigmas,smoothed_sigmas,predicted_sigmas;

        Naive_RTS_Smoother(const State &_mu, const Covariance &_sigma) : ADEKF<State>(_mu, _sigma) { storeEstimation(); storePredictedEstimation(); }
        Naive_RTS_Smoother(const ADEKF<State> &filter) : ADEKF<State>(filter) { storeEstimation();storePredictedEstimation(); }
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
    //Taken from https://gist.github.com/gokhansolak/d2abaefcf3e3b767f5bc7d81cfe0b36a
    template<typename _Matrix_Type_>
    _Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
    {
	Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothSingleStep(size_t k, DynamicModel & dynamicModel, const SquareMatrixType<double, NoiseDim> &Q,const std::tuple<Controls ...> & controls,const State & smoothed_mu_kplus,const Covariance & smoothed_sigma_kplus){
                MatrixType<double, DOF, DOF> Fk(DOF, DOF);
                Covariance SmootherGain;
                

                auto derivator = adekf::getDerivator<DOF>();
                auto input = eval(old_mus[k] + derivator);
                applyDynamicModel(dynamicModel,input,Eigen::Matrix<double, NoiseDim, 1>::Zero(), controls);
                auto result1 = eval(input - predicted_mus[k+1]);
                Eigen::Matrix<double, DOF, 1> diff1;

                for (size_t j = 0; j < DOF; ++j)
                {
                    Fk.row(j) = result1(j).v;
                    diff1(j) = result1(j).a;
                }

                SmootherGain = old_sigmas[k] * Fk.transpose() * (predicted_sigmas[k+1].inverse());
                smoothed_mus[k] = old_mus[k] + (SmootherGain * (smoothed_mu_kplus- predicted_mus[k+1]));
                smoothed_sigmas[k] = (old_sigmas[k] + SmootherGain *(smoothed_sigma_kplus- predicted_sigmas[k+1]) * SmootherGain.transpose());
                assurePositiveDefinite(smoothed_sigmas[k]);
        }


        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothIntervalWithNonAdditiveNoise(size_t steps, int start, DynamicModel & dynamicModel, const SquareMatrixType<double, NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
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
                smoothSingleStep(k,dynamicModel,Q,all_controls[k],smoothed_mus[k+1], smoothed_sigmas[k+1]);
                
            }
        }

        template <int NoiseDim, typename DynamicModel, typename... Controls>
        void smoothAllWithNonAdditiveNoise(DynamicModel & dynamicModel, const SquareMatrixType<double, NoiseDim> &Q, const std::vector<std::tuple<Controls...>> &all_controls)
        {
            smoothIntervalWithNonAdditiveNoise(old_mus.size() - 2, -1, dynamicModel, Q, all_controls);
        }
    };

    /**
     * General Deduction Template for the Naive_RTS_Smoother based on StateRetriever.
     * This is needed so you can type Naive_RTS_Smoother smoother(Filter{...}) without template arguments
     */
    /*template <typename State>
    Naive_RTS_Smoother(ADEKF<State> &&) -> Naive_RTS_Smoother<State>;
    template <typename State>
    Naive_RTS_Smoother(ADEKF<State> ) -> Naive_RTS_Smoother<State>;
*/

} // namespace adekf