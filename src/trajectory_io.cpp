#include "realtime_tracking/trajectory_io.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>

void write_trajectory_csv(
    const std::vector<State2D>& trajectory,
    const std::filesystem::path& output_path)
{
    std::ofstream output{output_path};

    if (!output)
    {
        throw std::runtime_error{
            "Could not open trajectory output file."
        };
    }

    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "step,x,y,vx,vy\n";

    for (std::size_t step{0};
         step < trajectory.size();
         ++step)
    {
        const State2D& state = trajectory[step];

        output
            << step << ','
            << state.x << ','
            << state.y << ','
            << state.vx << ','
            << state.vy << '\n';
    }

    output.close();

    if (!output)
    {
        throw std::runtime_error{
            "Failed while writing trajectory output file."
        };
    }
}
