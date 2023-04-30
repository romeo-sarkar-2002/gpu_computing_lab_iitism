CC = nvcc
FLAGS = -dc -c


run: bin/main.exe
	bin/main.exe

bin/main.exe: lib/main.o lib/arrayManip.o lib/sumReduction.o lib/sumReduction_lessWarpDivergence.o
	$(CC) lib/main.o lib/arrayManip.o lib/sumReduction.o lib/sumReduction_lessWarpDivergence.o -o bin/main

lib/main.o: main.cu
	$(CC) $(FLAGS) main.cu -o "lib/main.o"
lib/arrayManip.o: sources/arrayManip.cu
	$(CC) $(FLAGS) sources/arrayManip.cu -o "lib/arrayManip.o"
lib/sumReduction.o: sources/sumReduction.cu
	$(CC) $(FLAGS) sources/sumReduction.cu -o "lib/sumReduction.o"
lib/sumReduction_lessWarpDivergence.o: sources/sumReduction_lessWarpDivergence.cu
	$(CC) $(FLAGS) sources/sumReduction_lessWarpDivergence.cu -o "lib/sumReduction_lessWarpDivergence.o"


clean:
	del "lib/*.o"
	del "bin/main*"