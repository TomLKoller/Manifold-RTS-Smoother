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
        static constexpr size_t DOF = DOFOf<State>;
        static constexpr double weighted_mean_epsilon = 1e-8;
        aligned_vector<State> old_mus,smoothed_mus, predicted_mus;
        aligned_vector<Covariance> old_sigmas,smoothed_sigmas,predicted_sigmas;

        RTS_Smoother(const State &_mu, const Covariance &_sigma) : ADEKF<State>(_mu, _sigma) { storeEstimation(); storePredictedEstimation(); }
        RTS_Smoother(const ADEKF<State> &filter) : ADEKF<State>(filter) { storeEstimation();storePredictedEstimation(); }
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
            State smoothed = old_mus[k];
                MatrixType<double, DOF, DOF> Fk(DOF, DOF);
                Covariance SmootherGain;
                

                auto derivator = adekf::getDerivator<DOF>();
                auto input = eval(smoothed + derivator);
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
                //Calc Covariance
                /*SquareMatrixType<double,2*DOF> measCovariance(2*DOF,2*DOF);
                Covariance DynamicCovariance=getNonAdditiveDynamicNoise(old_mus[k],dynamicModel,Q,controls);
                measCovariance.template block<DOF,DOF>(0,0)=old_sigmas[k];
                measCovariance.template block<DOF,DOF>(0,DOF)=old_sigmas[k]*J1.transpose();
                measCovariance.template block<DOF,DOF>(DOF,0)=J1*old_sigmas[k].transpose();
                measCovariance.template block<DOF,DOF>(DOF,DOF)=smoothed_sigma_kplus+DynamicCovariance;
                Eigen::Matrix<ceres::Jet<double,DOF>,2*DOF,1> result2;
                result2.template segment<DOF>(0) = eval(smoothed_mus[k]+derivator - old_mus[k]);
                input=smoothed_mus[k]+derivator;
                applyDynamicModel(dynamicModel,input,Eigen::Matrix<double, NoiseDim, 1>::Zero(), controls);
                result2.template segment<DOF>(DOF) = eval(input - smoothed_mu_kplus);


                MatrixType<double, 2*DOF,DOF> J2(2*DOF,DOF);
                for (size_t j = 0; j < 2*DOF; ++j)
                {
                    J2.row(j) = result2(j).v;
                }
                std::cout << measCovariance.determinant() << std::endl;
                smoothed_sigmas[k]=(J2.transpose()*measCovariance.inverse()*J2).inverse();*/


                Covariance Jk(DOF,DOF);
                //derived
                auto result2=eval(old_mus[k]+(SmootherGain * (smoothed_mu_kplus- predicted_mus[k+1])+derivator)-smoothed_mus[k]);
                //works better
                //result2=eval(smoothed_mus[k]+derivator-old_mus[k]);
                for (size_t j = 0; j < DOF; ++j)
                {
                    Jk.row(j) = result2(j).v;
                }
                //Jk=Jk.Identity(DOF,DOF);
                Covariance Dk(DOF,DOF);
                result2=eval((old_mus[k] + (SmootherGain * (smoothed_mu_kplus +derivator- predicted_mus[k+1])))-smoothed_mus[k]);
                result2=eval((old_mus[k] + (SmootherGain * (smoothed_mu_kplus +derivator- predicted_mus[k+1])))-old_mus[k]);
                for (size_t j = 0; j < DOF; ++j)
                {
                    Dk.row(j) = result2(j).v;
                }
                //50.5971
                //std::cout << (SmootherGain-D).norm() << std::endl;
                //smoothed_sigmas[k] = J2 * (old_sigmas[k] + SmootherGain * (smoothed_sigma_kplus - predicted_sigmas[k+1]) * SmootherGain.transpose()) * J2.transpose();
                smoothed_sigmas[k] = Jk*(old_sigmas[k] - SmootherGain * predicted_sigmas[k+1] * SmootherGain.transpose()+Dk*smoothed_sigma_kplus*Dk.transpose())*Jk.transpose();
                //smoothed_sigmas[k] = Jk*(old_sigmas[k] - SmootherGain * predicted_sigmas[k+1] * SmootherGain.transpose())*Jk.transpose()+Dk*smoothed_sigma_kplus*Dk.transpose();
                 //assert(isPositiveDefinite(smoothed_sigmas[k]));
                assurePositiveDefinite(smoothed_sigmas[k]);
                //assert(isPositiveDefinite(smoothed_sigmas[k]));
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
     * General Deduction Template for the RTS_Smoother based on StateRetriever.
     * This is needed so you can type RTS_Smoother smoother(Filter{...}) without template arguments
     */
    /*template <typename State>
    RTS_Smoother(ADEKF<State> &&) -> RTS_Smoother<State>;
    template <typename State>
    RTS_Smoother(ADEKF<State> ) -> RTS_Smoother<State>;
*/

} // namespace adekf