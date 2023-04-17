#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

using uint = unsigned int;

std::vector<uint> ReadVectorFromFile(const std::string &filename) 
{
    std::ifstream infile(filename.c_str());

    if (!infile) 
        throw std::runtime_error("Could not open file!");

    std::vector<uint> res;
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        res.push_back(static_cast<uint>(std::stoul(line)));
    }

    return res;
}

void WriteVectorToFile(const std::string &filename, const std::vector<uint> &v) 
{
    std::ofstream outfile(filename.c_str());

    if (!outfile.is_open()) 
        throw std::runtime_error("Could not open file!");

    for (auto element : v)
        outfile << element << std::endl;

    outfile.close();
}


#endif 