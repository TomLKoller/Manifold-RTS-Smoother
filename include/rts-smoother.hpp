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

        RTS_Smoother(const State &_mu, const Covariance &_sigma): ADEKF<State>(_mu,_sigma){}
        RTS_Smoother(const ADEKF<State> &filter): ADEKF<State>(filter){}
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
template<int NoiseDim, typename DynamicModel, typename ... Controls>
Covariance getNonAdditiveDynamicNoise(const State & smoothed, DynamicModel dynamicModel, const SquareMatrixType<double,NoiseDim> &Q,const Controls & ... u){
     auto derivator=adekf::getDerivator<DOF+NoiseDim>();
                    auto input=eval(smoothed +derivator.template head<DOF>());
                    //Todo u vernünftig machen
                    dynamicModel(input,derivator.template tail<NoiseDim>() , u...);
                    auto result = (input-smoothed);
                    //std::cout << result << std::endl;
                    Eigen::Matrix<double,DOF,DOF+NoiseDim> F(DOF,DOF+NoiseDim);
                    //Eigen::Matrix<double,DOF,1> diff;
                    for (size_t j = 0; j < DOF; ++j)
                    {
                        F.row(j) = result(j).v;
                        //diff(j)=result(j).a;
                    }
                    return eval(F.template rightCols<NoiseDim>()*Q*F.template rightCols<NoiseDim>().transpose());
}


        template <int NoiseDim,typename DynamicModel, typename... Controls>
        void smoothIntervalWithNonAdditiveNoise(size_t steps, int start, DynamicModel dynamicModel, const SquareMatrixType<double,NoiseDim> &Q, const Controls &...u)
        {
            if(start <0){
                start=old_mus.size()-1+start;
            }
            assert(start-steps >=0 && "Can not smooth more estimates than available");
            assert(start < old_mus.size() - 1 && "Can not smooth a state which belongs to future");
            for (size_t i = start; i >= old_mus.size() - steps - 1; i--)
            {
                std::cout << "At iteration: "<< i <<std::endl;
                State smoothed = old_mus[i];
                int counter=0;
                ADEKF_MINUSRESULT(State,State) optimize, o1, o2;
                 Eigen::Matrix<double,DOF,DOF> J1(DOF,DOF),J2(DOF,DOF);
                 Covariance DynamicCov, PredictedCov;
                do
                {
                    DynamicCov=getNonAdditiveDynamicNoise(smoothed,dynamicModel,Q,u ...);

                    auto derivator1=adekf::getDerivator<DOF>();
                    auto input1=eval(smoothed +derivator1);
                    //Todo u vernünftig machen
                    dynamicModel(input1,Eigen::Matrix<double,NoiseDim,1>::Zero() , u...);
                     auto result1=eval(smoothed_mus[i+1]-input1);
                    Eigen::Matrix<double,DOF,1> diff1;
                   
                   
                    for (size_t j = 0; j < DOF; ++j)
                    {
                        J1.row(j) = result1(j).v;
                        diff1(j)=result1(j).a;
                    }

                    auto result2=eval(smoothed+adekf::getDerivator<DOF>()-old_mus[i]);
                    Eigen::Matrix<double,DOF,1> diff2;
                   
                    for (size_t j = 0; j < DOF; ++j)
                    {
                        J2.row(j) = result2(j).v;
                        diff2(j)=result2(j).a;
                    }
                    
                    //std::cout << DynamicCov.inverse() <<std::endl;
                    //std::cout << DynamicCov<<std::endl;
                    //std::cout << diff <<std::endl;
                    //std::cout << J1 <<std::endl<<std::endl;
                    //std::cout << J1.template leftCols<DOF>().inverse() <<std::endl<<std::endl;
                    //std::cout << J1.template leftCols<DOF>().transpose()*DynamicCov.inverse()*(diff) <<std::endl;
                    //std::cout << J2.transpose()*old_sigmas[i].inverse()*(diff2) << std::endl<< std::endl;
                     //optimize=2*(J1.template leftCols<DOF>().transpose()*DynamicCov.inverse()*(diff)+J2.transpose()*old_sigmas[i].inverse()*(diff2));
                    //optimize=((diff1)+DynamicCov*J1.transpose().inverse()*J2.transpose()*old_sigmas[i].inverse()*(diff2));
                   
                   
                   PredictedCov=J1*old_sigmas[i]*J1.transpose() +DynamicCov;
                   //optimize=J1.transpose()*PredictedCov.inverse()*diff1+J2.transpose()*old_sigmas[i].inverse()*diff2-J1.transpose()*PredictedCov.inverse()*J1*old_sigmas[i]*J2.transpose()*old_sigmas[i].inverse()*diff2;
                   //std::cout << PredictedCov << std::endl;
                   //std::cout << J1 << std::endl;
                   State smoothCopy=smoothed;
                   dynamicModel(smoothCopy,Eigen::Matrix<double,NoiseDim,1>::Zero() , u...);
                   optimize=(DynamicCov*J1.transpose().inverse()*J2.transpose()*old_sigmas[i].inverse()*diff2);
                   optimize=(smoothCopy+optimize)-smoothed_mus[i+1];
                   //std::cout << "old_sigma \n" << old_sigmas[i] << std::endl;
                   //auto test=old_sigmas[i]*J2.transpose().inverse();
                   //o1=(old_sigmas[i]*J2.transpose().inverse()*(diff2.dot(diff1)) +(diff1.dot(diff1))*DynamicCov*J1.transpose().inverse())*diff1;
                   //o2=(old_sigmas[i]*J2.transpose().inverse()*(diff2.dot(diff2)) +(diff1.dot(diff2))*DynamicCov*J1.transpose().inverse())*diff2;
                   //optimize=o1+o2;
                   //std::cout << smoothed << std::endl;    
                    smoothed=smoothed+optimize;
                   // optimize=(old_sigmas[i]*J2.inverse().transpose()*J1.transpose()*DynamicCov.inverse()*diff1);
                   //smoothed=old_mus[i]+optimize;
                    //std::cout << optimize.transpose() << std::endl;
                    //std::cout << smoothed << std::endl;
                } while (optimize.norm() > weighted_mean_epsilon  && ++counter <150 );
                //} while ( ++counter <10 );
                std::cout << smoothed <<std::endl;
                //std::cout << optimize.transpose() <<std::endl;
                smoothed_mus[i]=smoothed;
                //Todo this is wrong
                //smoothed_sigmas[i]=J1.transpose()*DynamicCov*J1+J2.transpose()*old_sigmas[i]*J2;
                //std::cout<<(smoothed_sigmas[i].determinant() !=0.) << std::endl;           
                }
        }

        template <int NoiseDim,typename DynamicModel, typename... Controls>
        void smoothAllWithNonAdditiveNoise(DynamicModel dynamicModel, const SquareMatrixType<double,NoiseDim> &Q, const Controls &...u)
        {
            smoothIntervalWithNonAdditiveNoise(old_mus.size() - 2,-1, dynamicModel, Q, u...);
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