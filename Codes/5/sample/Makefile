NVCC = nvcc.exe
# NVCC_FLAGS = -g -G -Xcompiler -Wall
# all: distanceMain.exe
# distanceMain.exe: distanceMain.o DistKernel.o
# 	$(NVCC) $^ -o $@
# distanceMain.o: distanceMain.cpp DistKernel.h
# 	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
# Distkernel.o: DistKernel.cu DistKernel.h
# 	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

# echo hello world
all: distanceMain.exe
distanceMain.exe: distanceMain.obj DistKernel.obj
	$(NVCC) $^ -o $@

distanceMain.obj: distanceMain.cpp
	$(NVCC) -c $^ -o $@

DistKernel.obj: DistKernel.cu
	$(NVCC) -c $^ -o $@