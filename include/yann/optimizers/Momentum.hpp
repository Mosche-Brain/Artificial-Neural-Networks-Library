/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : src/optimizers/Momentum.cpp
 * @createdOn   : 09/08/2026
 * @description : Declaration of SGD with momentum optimizer class
 *---------------------------------------------**/
#pragma once

#include <unordered_map>                    
                                   
#include "yann/optimizers/OptimizerBase.hpp"
                                   
namespace yann::optimizers         
{                                  
    class Momentum : public OptimizerBase
    {                              
    public:                        
        Momentum(cum::cumeric_t rate, cum::cumeric_t beta = 0.5);
                                                                                                                  
        void step(Parameter& params) override;
                                                               
        void step(std::vector<Parameter*>& params) override;                                                       
                                                                                                                  
        std::unique_ptr<Momentum> create(cum::cumeric_t rate, cum::cumeric_t beta = 0.5);                               
                                                                                                   
    private:                                                                                       
        cum::cumeric_t b;                                                                          
        std::unordered_map<Parameter*, cum::Matrix> momentum;                                      
    };                                                                                                                                                    
}                                                                                                                                                         
                                                                                                                                                               
                                                                                                                                                               
                
