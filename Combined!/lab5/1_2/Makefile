CC = nvcc
FLAGS = -dc -c


run: bin/main.exe
	bin/main.exe

bin/main.exe: lib/main.o lib/matrix.o matrix.cuh
	$(CC) lib/main.o lib/matrix.o -o bin/main.o

lib/main.o: main.cu matrix.cuh
	$(CC) $(FLAGS) main.cu -o "lib/main.o"
lib/matrix.o: matrix.cu matrix.cuh
	$(CC) $(FLAGS) matrix.cu -o "lib/matrix.o"


CLEAN:
	del "lib/*.o"
	del "bin/main*"
