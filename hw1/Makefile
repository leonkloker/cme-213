CXX=g++
CXXFLAGS=-g -Wall -pthread -std=c++14 -O
SRC1=main_q1.cpp
SRC2=main_q2.cpp
SRC3=main_q3.cpp
SRC4=main_q4.cpp
SRC5=main_q5.cpp
HDR=matrix.hpp
HDR2= matrix_rect.hpp

# Points to the root of Google Test, relative to where this file is.
GTEST_DIR=./googletest-main/googletest
CPPFLAGS += -isystem $(GTEST_DIR)/include

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)


default: main_q1 main_q2 main_q3 main_q4 main_q5

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

# Build our homework code
main_q1.o : $(SRC1) $(HDR) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC1)
main_q1: main_q1.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@
main_q2.o : $(SRC2) $(HDR2) $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC2)
main_q2: main_q2.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@
main_q3: $(SRC3)
	$(CXX) $(CXXFLAGS) $< -o $@
main_q4: $(SRC4) 
	$(CXX) $(CXXFLAGS) $< -o $@
main_q5: $(SRC5) 
	$(CXX) $(CXXFLAGS) $< -o $@

run: main_q1 main_q2 main_q3 main_q4 main_q5
	./main_q1
	./main_q2
	./main_q3
	./main_q4
	./main_q5

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q3 main_q4 main_q5
	rm -rf *.dSYM
	rm -rf gtest.a gtest_main.a

