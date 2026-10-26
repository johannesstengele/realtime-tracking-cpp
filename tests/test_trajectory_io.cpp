#include "realtime_tracking/state2d.hpp"
#include "realtime_tracking/trajectory_io.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
class CommaDecimalPoint : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override
    {
        return ',';
    }
};
}

int main()
{
    int failures{};
    const std::filesystem::path directory{"trajectory_io_test"};

    if (!std::filesystem::create_directory(directory))
    {
        std::cerr << "Temporary directory already exists.\n";
        return 1;
    }

    const auto output_path = directory / "trajectory.csv";
    const auto empty_path = directory / "empty.csv";
    const std::vector<State2D> trajectory{
        {1.25, -2.5, 0.125, -0.25},
        {std::numeric_limits<double>::max(),
         std::numeric_limits<double>::min(),
         std::numeric_limits<double>::denorm_min(),
         1.2345678901234567}
    };

    const std::locale original_locale = std::locale();
    std::locale::global(std::locale{std::locale::classic(), new CommaDecimalPoint});

    try
    {
        write_trajectory_csv(trajectory, output_path);
        write_trajectory_csv({}, empty_path);
    }
    catch (...)
    {
        std::locale::global(original_locale);
        throw;
    }

    std::locale::global(original_locale);

    {
        std::ifstream input{output_path};
        std::string line;
        std::getline(input, line);

        if (line != "step,x,y,vx,vy")
        {
            std::cerr << "Incorrect CSV header.\n";
            ++failures;
        }

        for (std::size_t index{}; index < trajectory.size(); ++index)
        {
            if (!std::getline(input, line))
            {
                std::cerr << "Missing CSV row.\n";
                ++failures;
                continue;
            }

            std::istringstream row{line};
            row.imbue(std::locale::classic());
            std::size_t step{};
            State2D state{};
            char separators[4]{};

            if (!(row >> step >> separators[0]
                      >> state.x >> separators[1]
                      >> state.y >> separators[2]
                      >> state.vx >> separators[3] >> state.vy)
                || step != index
                || state.x != trajectory[index].x
                || state.y != trajectory[index].y
                || state.vx != trajectory[index].vx
                || state.vy != trajectory[index].vy
                || std::string(separators, 4) != ",,,,")
            {
                std::cerr << "CSV row did not round-trip: " << line << '\n';
                ++failures;
            }

            row >> std::ws;

            if (!row.eof())
            {
                std::cerr << "Unexpected CSV fields.\n";
                ++failures;
            }
        }

        if (std::getline(input, line))
        {
            std::cerr << "Unexpected CSV row.\n";
            ++failures;
        }
    }

    {
        std::ifstream input{empty_path};
        std::string header;
        std::string extra;

        if (!std::getline(input, header)
            || header != "step,x,y,vx,vy"
            || std::getline(input, extra))
        {
            std::cerr << "Empty trajectory must produce only the CSV header.\n";
            ++failures;
        }
    }

    for (const auto& invalid_path : {
             directory,
             directory / "missing" / "trajectory.csv"})
    {
        bool rejected{false};

        try
        {
            write_trajectory_csv(trajectory, invalid_path);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }

        if (!rejected)
        {
            std::cerr << "Accepted invalid CSV destination.\n";
            ++failures;
        }
    }

    std::filesystem::remove(output_path);
    std::filesystem::remove(empty_path);
    std::filesystem::remove(directory);

    return (failures == 0) ? 0 : 1;
}
