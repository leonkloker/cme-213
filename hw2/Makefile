CXX=g++
CXXFLAGS=-std=c++14 -g -Wall -O3 -fopenmp
SRC1=main_q1.cpp
SRC2=main_q2.cpp
HDR1=sum.h
HDR2=parallel_radix_sort.h

# Points to the root of Google Test, relative to where this file is.
GTEST_DIR=./googletest-main/googletest
CPPFLAGS += -isystem $(GTEST_DIR)/include

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

default: main_q1 main_q2 main_q2_part6

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

main_q1.o : $(SRC1) $(HDR) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC1)
main_q1: main_q1.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2)
main_q2: main_q2.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

main_q2_part6.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -D QUESTION6 -c $(SRC2) -o main_q2_part6.o
main_q2_part6: main_q2_part6.o gtest_main.a
	$(CXX) $(CXXFLAGS) -lpthread $^ -D QUESTION6 -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q2_part6
	rm -rf *.dSYM
	rm -rf gtest_main.a
