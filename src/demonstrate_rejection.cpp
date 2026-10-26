#include "realtime_tracking/demonstrate_rejection.hpp"
#include "realtime_tracking/constant_velocity_model.hpp"

#include <iostream>
#include <stdexcept>

void demonstrate_rejection(double dt){
    try
    {
        static_cast<void>(ConstantVelocityModel{dt});
    }
    catch(const std::invalid_argument& e)
    {
        std::cerr << e.what() << '\n';
    }
}
