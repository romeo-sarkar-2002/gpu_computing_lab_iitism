NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
PROG_NAME = implementation

all: build

run: build
	./$(PROG_NAME)

build: $(PROG_NAME)

$(PROG_NAME): init.o $(PROG_NAME).o
	$(NVCC) $^ -o $@

init.o: init.cpp $(PROG_NAME).h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(PROG_NAME).o: $(PROG_NAME).cu $(PROG_NAME).h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean: clean_temp
	rm -f $(PROG_NAME)

clean_temp:
	rm -f *.o
