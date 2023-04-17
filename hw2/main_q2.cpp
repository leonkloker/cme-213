#include <iostream>
#include <random>
#include <cstdio>
#include <cassert>
#include <omp.h>

#include "test_macros.h"
#include "test_util.h"
#include "parallel_radix_sort.h"
#include "gtest/gtest.h"

constexpr const uint kSizeMaskTest = 8;
constexpr const uint kStartBitTest = 0;

void radixSortSerialPass(
    std::vector<uint> &keys, 
    std::vector<uint> &keys_radix,
    uint startBit, 
    uint numBits
) {
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;

    // compute the frequency histogram
    std::vector<uint> histogramRadixFrequency(numBuckets);

    for (uint i = 0; i < keys.size(); ++i)
    {
        uint key = (keys[i] >> startBit) & mask;
        ++histogramRadixFrequency[key];
    }

    // now scan it
    std::vector<uint> exScanHisto(numBuckets, 0);

    for (uint i = 1; i < numBuckets; ++i)
    {
        exScanHisto[i] = exScanHisto[i - 1] + histogramRadixFrequency[i - 1];
        histogramRadixFrequency[i - 1] = 0;
    }

    histogramRadixFrequency[numBuckets - 1] = 0;

    // now add the local to the global and scatter the result
    for (uint i = 0; i < keys.size(); ++i)
    {
        uint key = (keys[i] >> startBit) & mask;

        uint localOffset = histogramRadixFrequency[key]++;
        uint globalOffset = exScanHisto[key] + localOffset;

        keys_radix[globalOffset] = keys[i];
    }
}

int radixSortSerial(
    std::vector<uint> &keys, 
    std::vector<uint> &keys_radix,
    uint numBits
) {
    assert(numBits <= 16);

    for (uint startBit = 0; startBit < 32; startBit += 2 * numBits)
    {
        radixSortSerialPass(keys, keys_radix, startBit, numBits);
        radixSortSerialPass(keys_radix, keys, startBit + numBits, numBits);
    }

    return 0;
}

void initializeRandomly(std::vector<uint> &keys)
{
    std::default_random_engine generator;
    std::uniform_int_distribution<uint> distribution(0, kRandMax);

    for (uint i = 0; i < keys.size(); ++i)
        keys[i] = distribution(generator);
}

TEST(testQ2, test1)
{
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/blockhistograms");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> blockHistograms = computeBlockHistograms(input, numBlocks,
                                        numBuckets, kSizeMaskTest, kStartBitTest, blockSize);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockHistograms, &success);
}

TEST(testQ2, test2)
{
    std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/globalhisto");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms,
                                    numBlocks, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHisto, &success);
}

TEST(testQ2, test3)
{
    std::vector<uint> globalHisto = ReadVectorFromFile("test_files/globalhisto");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/globalhistoexscan");

    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, globalHistoExScan, &success);
}

TEST(testQ2, test4)
{
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/blockexscan");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> globalHistoExScan = ReadVectorFromFile("test_files/globalhistoexscan");
    std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets,
                                    numBlocks, globalHistoExScan, blockHistograms);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, blockExScan, &success);
}

TEST(testQ2, test5)
{
    std::vector<uint> blockExScan = ReadVectorFromFile("test_files/blockexscan");
    std::vector<uint> input = ReadVectorFromFile("test_files/input");
    std::vector<uint> expected_output = ReadVectorFromFile("test_files/sorted");

    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << kSizeMaskTest;
    std::vector<uint> sorted(input.size());
    populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, kStartBitTest,
                                  kSizeMaskTest, blockSize, input, sorted);
    bool success = true;
    EXPECT_VECTOR_EQ(expected_output, sorted, &success);
}

TEST(testQ2, serialSortTest)
{
    // Initialize Variables
    std::vector<uint> keys_stl(kSizeTestVector);
    initializeRandomly(keys_stl);
    std::vector<uint> keys_serial = keys_stl;
    std::vector<uint> keys_parallel = keys_stl;
    std::vector<uint> temp_keys(kSizeTestVector);

    // stl sort
    double startstl = omp_get_wtime();
    std::sort(keys_stl.begin(), keys_stl.end());
    double endstl = omp_get_wtime();

    // serial radix sort
    double startRadixSerial = omp_get_wtime();
    radixSortSerial(keys_serial, temp_keys, kNumBits);
    double endRadixSerial = omp_get_wtime();

    bool success = true;
    EXPECT_VECTOR_EQ(keys_stl, keys_serial, &success);
    std::cout << "Serial Radix Sort: " << ((success) ? "PASS" : "FAIL") << std::endl;

    std::cout << "stl: " << endstl - startstl << std::endl;
    std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;
}

TEST(testQ2, parallelSortTest)
{
    // Initialize Variables
    std::vector<uint> keys_stl(kSizeTestVector);
    initializeRandomly(keys_stl);
    std::vector<uint> keys_serial = keys_stl;
    std::vector<uint> keys_parallel = keys_stl;
    std::vector<uint> temp_keys(kSizeTestVector);

    // stl sort
    double startstl = omp_get_wtime();
    std::sort(keys_stl.begin(), keys_stl.end());
    double endstl = omp_get_wtime();

    // parallel radix sort
    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys_parallel, temp_keys, kNumBits, 8);
    double endRadixParallel = omp_get_wtime();

    bool success = true;
    EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);
    std::cout << "Parallel Radix Sort: " << ((success) ? "PASS" : "FAIL") << std::endl;

    std::cout << "stl: " << endstl - startstl << std::endl;
    std::cout << "parallel radix: " << endRadixParallel - startRadixParallel << std::endl;
}

#ifdef QUESTION6
TEST(testQ2, radixSortTest)
{
    // Initialize Variables
    std::vector<uint> keys_stl(kSizeTestVector);
    initializeRandomly(keys_stl);
    std::vector<uint> keys_parallel = keys_stl;
    std::vector<uint> temp_keys(kSizeTestVector);

    std::vector<uint> keys_orig = keys_stl;

    // stl sort
    std::sort(keys_stl.begin(), keys_stl.end());

    std::vector<uint> jNumBlock = {1, 2, 4, 8, 12, 16, 24, 32, 40, 48};

    printf("Threads Blocks / Timing\n  ");
    for (auto jNum : jNumBlock)
    {
        printf("%8d", jNum);
    }
    printf("\n");

    bool success = true;

    for (auto n_threads : jNumBlock)
    {
        printf("%4d ", n_threads);
        for (auto jNum : jNumBlock)
        {
            keys_parallel = keys_orig;
            runBenchmark(keys_parallel, temp_keys, kNumBits, jNum, n_threads);
            EXPECT_VECTOR_EQ(keys_stl, keys_parallel, &success);
        }
        printf("\n");
    }
}
#endif
