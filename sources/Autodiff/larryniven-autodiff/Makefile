CXXFLAGS += -std=c++11 -I ../
NVCCFLAGS += -std=c++11 -I ../
AR = gcc-ar

.PHONY: all clean gpu

all: libautodiff.a

gpu: libautodiffgpu.a

clean:
	-rm *.o
	-rm libautodiff.a libautodiffgpu.a

libautodiff.a: autodiff.o autodiff-op.o
	$(AR) rcs $@ $^

libautodiffgpu.a: autodiff.o autodiff-op.o autodiff-gpu.o autodiff-op-gpu.o
	$(AR) rcs $@ $^

autodiff-gpu.o: autodiff-gpu.cu
	nvcc $(NVCCFLAGS) -c autodiff-gpu.cu

autodiff-op-gpu.o: autodiff-op-gpu.cu
	nvcc $(NVCCFLAGS) -c autodiff-op-gpu.cu

autodiff.o: autodiff.h
autodiff-op.o: autodiff-op.h
autodiff-gpu.o: autodiff-gpu.h
autodiff-op-gpu.o: autodiff-op-gpu.h

