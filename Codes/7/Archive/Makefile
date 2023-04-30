NVCC =  /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

MAIN = lab7ex1main
LIB = lab7ex1lib
H = cuh

all: a.out

a.out: $(MAIN).o $(LIB).o
	echo "making..."
	$(NVCC) $^ -o $@

$(MAIN).o: $(MAIN).cu $(LIB).$(H)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

$(LIB).o: $(LIB).cu $(LIB).$(H)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

run: a.out
	echo "running..."
	./a.out

clean:
	echo "cleaning..."
	rm -rf *.o a.out