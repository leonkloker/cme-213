CXX=nvcc
CXXFLAGS=-O3 -std=c++14 -arch=compute_75 -code=sm_75
SRC1=main_q1.cu
SRC2=main_q2.cu
SRC3=main_q3.cu
HDR1=recurrence.cuh benchmark.cuh test_recurrence.h
HDR2=pagerank.cuh benchmark.cuh
HDR3=util.cuh benchmark.cuh
INC=.

# Points to the root of Google Test, relative to where this file is.
GTEST_DIR=./googletest-main/googletest
CPPFLAGS += -isystem $(GTEST_DIR)/include

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

.PHONY: default run3 clean

default: main_q3

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

main_q1.o : $(SRC1) $(HDR1) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC1)
main_q1: main_q1.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2)
main_q2: main_q2.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q3.o : $(SRC3) $(HDR3) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC3)
main_q3: main_q3.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

run1: main_q1
	srun --partition=CME --gres=gpu:1 ./main_q1

run2: main_q2
	srun --partition=CME --gres=gpu:1 ./main_q2

run3: main_q3
	srun --partition=CME --gres=gpu:1 ./main_q3

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q3
	rm -rf *.dSYM
