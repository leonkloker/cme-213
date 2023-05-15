#ifndef ERRORS_H_
#define ERRORS_H_

#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include "mp1-util.h"
#include "Grid.h"
#include "simParams.h"
#include "gtest/gtest.h"

#define PRINT_ERR

using std::cout;
using std::endl;
using std::setprecision;
using std::setw;

int checkErrors(const Grid &ref_grid, const Grid &gpu_grid,
                const simParams &params, std::string filename, std::vector<double> &errors)
{
    // check that we got the same answer
    std::ofstream ofs(filename.c_str());
    int error = 0;
    double l2ref = 0;
    double linf = 0;
    double l2err = 0;

    for (int x = 0; x < params.gx(); ++x)
    {
        for (int y = 0; y < params.gy(); ++y)
        {
            const double ref = ref_grid.hGrid_[x + params.gx() * y];
            const double gpu = gpu_grid.hGrid_[x + params.gx() * y];
            if (!AlmostEqualUlps(ref, gpu, 512))
            {
                ofs << "Mismatch at pos (" << x << ", " << y << ") cpu: "
                    << ref << " gpu: " << gpu << endl;
                ++error;
            }

            l2ref += ref * ref;
            l2err += (ref - gpu) * (ref - gpu);

            if (ref != 0)
                linf = max(abs(ref - gpu), linf);
        }
    }

    l2err = sqrt(l2err / params.gx() / params.gy());
    l2ref = sqrt(l2ref / params.gx() / params.gy());

#ifdef PRINT_ERR
    if (error)
        std::cerr << "There were " << error
                  << " total locations where there was a difference between the cpu and gpu" << endl;
#endif

    errors.push_back(l2ref);
    errors.push_back(linf);
    errors.push_back(l2err);

    ofs.close();

    return error;
}

void PrintErrors(const std::vector<double> &errors)
{
    cout << endl;
    cout << setw(15) << "L2Ref" << setw(15) << "LInf" << setw(15) << "L2Err" << endl;

    if (errors.size() > 0)
    {
        cout << setw(15) << setprecision(6) << errors[0]
             << setw(15) << errors[1] << setw(15) << errors[2] << endl;
    }
    cout << endl;
}

#endif
