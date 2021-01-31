CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../la
NVCCFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../la

.PHONY: all gpu clean

all: test-autodiff

gpu: test-autodiff-gpu

clean:
	-rm *.o
	-rm test-autodiff test-autodiff-op-gpu test-autodiff-gpu

test-autodiff: test-autodiff.o ../autodiff.o ../autodiff-op.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lla -lebt -lblas

test-autodiff-gpu: test-autodiff-gpu.o ../autodiff.o ../autodiff-gpu.o ../autodiff-op.o ../autodiff-op-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -llagpu -lebt -lcublas -lcudart -lblas

test-autodiff-op-gpu:: test-autodiff-op-gpu.o ../autodiff-op-gpu.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -llagpu -lebt -lcublas -lcudart -lblas

test-autodiff-gpu.o: test-autodiff-gpu.cu
	nvcc $(NVCCFLAGS) -c test-autodiff-gpu.cu

test-autodiff-op-gpu.o: test-autodiff-op-gpu.cu
	nvcc $(NVCCFLAGS) -c test-autodiff-op-gpu.cu
