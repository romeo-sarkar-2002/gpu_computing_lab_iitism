NVCC =  /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

MAIN = main
LIB = lib
H = cuh

all: a.out

a.out: $(MAIN).o $(LIB).o
	echo "making..."
	$(NVCC) $^ -o $@

$(MAIN).o: $(MAIN).cu $(LIB).$(H)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(LIB).o: $(LIB).cu $(LIB).$(H)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: a.out
	echo "running..."
	./a.out

clean:
	echo "cleaning..."
	rm -rf *.o a.out