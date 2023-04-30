CC = nvcc
FLAGS = -dc -c

target: lib\main.obj lib\matrix.obj
	$(CC) lib\main.obj lib\matrix.obj -o bin\main
	bin\main.exe

lib\main.obj: main.cu
	$(CC) $(FLAGS) main.cu -o "lib\main"
lib\matrix.obj: matrix.cu
	$(CC) $(FLAGS) matrix.cu -o "lib\matrix"

CLEAN:
	del "lib\*.obj"
	del "bin\main*"
