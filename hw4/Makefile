# Points to the root of Google Test, relative to where this file is.
GTEST_ROOT=../hw3/googletest-main

GTEST_DIR=$(GTEST_ROOT)/googletest
GTEST_INC=$(GTEST_DIR)/include
GTEST_SRC=$(GTEST_DIR)/src

# Choose your option
# fmad=false will mean lower performance but the results will match the CPU results.
# No option (empty) means that the code runs faster but there will be roundoff 
# errors compared to the CPU.
# Make sure to run 
# $ make clean && make
# when changing this option so all the input files recompile correctly.
FMAD_FLAG=--fmad=false
# FMAD_FLAG=

CXX=g++
CUD=nvcc
LDFLAGS=-lcudart
CXXFLAGS=-O3 -Wall -Wextra -Wno-strict-aliasing -std=c++14 
CUDFLAGS=$(FMAD_FLAG) -O3 -arch=compute_75 -code=sm_75 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing -std=c++14
HDR=main.cu mp1-util.h simParams.h Grid.h gpuStencil.cu BC.h \
	CPUComputation.h Errors.h

CPPFLAGS += -isystem $(GTEST_INC)

GTEST_HEADERS = $(GTEST_INC)/gtest/*.h \
                $(GTEST_INC)/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_SRC)/*.cc $(GTEST_SRC)/*.h $(GTEST_HEADERS)

.PHONY: default clean

default: main

gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_SRC)/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

main: main.o simParams.o Grid.o gtest_main.a
	$(CUD) $^ $(LDFLAGS) $(CUDFLAGS) $(CPPFLAGS) -o $@

main.o: $(HDR) $(GTEST_HEADERS)
	$(CUD) -c $< $(CUDFLAGS) $(CPPFLAGS)

simParams.o: simParams.cpp simParams.h
	$(CXX) -c $< $(CXXFLAGS) -Winline

Grid.o: Grid.cu Grid.h
	$(CUD) -c $< $(CUDFLAGS)

clean:
	rm -f *.o *.a *~ *~ *Errors.txt main
