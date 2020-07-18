#include "../include/MDPLib.h"

#include <limits>

namespace mdplib
{
    double dead_end_cost = 10000.0;
    double epsilon = 1.0e-8;
    double no_distance = -std::numeric_limits<double>::max();
}
