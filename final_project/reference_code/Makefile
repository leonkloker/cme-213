OBJDIR=obj
LIBDIR=lib
INCDIR=inc

# Compilers
CC=mpic++
CUD=nvcc

# Flags
CFLAGS= -O2 -std=c++11 -I/usr/include # remove -02 and replace with -g for debug flags
#CFLAGS= -g -std=c++11 -I/usr/include # remove -02 and replace with -g for debug flags
LDFLAGS= -L/usr/local/cuda-10.2/lib64 -Wl,-rpath -Wl,.. -larmadillo -lcublas -lcudart 
CUDFLAGS= -O2 -c -arch=sm_37 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing 
#CUDFLAGS= -g -lineinfo -c -arch=sm_37 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing # remove -02 and replace with -g for debug flags. -lineinfo for nvprof. 
INCFLAGS= -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I$(INCDIR)
#-fmad=false

main: $(OBJDIR)/mnist.o $(OBJDIR)/tests.o $(OBJDIR)/common.o $(OBJDIR)/gpu_func.o $(OBJDIR)/neural_network.o $(OBJDIR)/main.o 
	cd $(OBJDIR); $(CC) main.o neural_network.o mnist.o common.o gpu_func.o tests.o -o ../main $(LDFLAGS) 

$(OBJDIR)/main.o: main.cpp utils/test_utils.h $(INCDIR)/neural_network.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCFLAGS) -c main.cpp -o $(OBJDIR)/main.o

$(OBJDIR)/neural_network.o: neural_network.cpp $(INCDIR)/neural_network.h utils/test_utils.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCFLAGS) -c neural_network.cpp -o $(OBJDIR)/neural_network.o

$(OBJDIR)/mnist.o: utils/mnist.cpp
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCFLAGS) -c utils/mnist.cpp -o $(OBJDIR)/mnist.o

$(OBJDIR)/tests.o: utils/tests.cpp utils/tests.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCFLAGS) -c utils/tests.cpp -o $(OBJDIR)/tests.o

$(OBJDIR)/common.o: utils/common.cpp
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(INCFLAGS) -c utils/common.cpp -o $(OBJDIR)/common.o

$(OBJDIR)/gpu_func.o: gpu_func.cu
	@mkdir -p $(OBJDIR)
	$(CUD) $(CUDFLAGS) $(INCFLAGS) -c gpu_func.cu -o $(OBJDIR)/gpu_func.o

clean:
	rm -rf $(OBJDIR)/*.o main

clear:
	rm -rf  cme213.* 
