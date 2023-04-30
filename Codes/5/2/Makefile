CC = nvcc
FLAGS = -dc -c

# Targets = Main.cu Matrix.cu
ALL: Lib\Main.obj Lib\Matrix.obj
	$(CC) Lib\Main.obj Lib\Matrix.obj -o Main
	.\Main.exe

Lib\Main.obj: Main.cu
	$(CC) $(FLAGS) Main.cu -o "Lib/Main"
Lib\Matrix.obj: Matrix.cu
	$(CC) $(FLAGS) Matrix.cu -o "Lib/Matrix"

CLEAN:
	del "Lib\*.obj"
	del "Main.exe"
	del "Main.lib"
	del "Main.exp"
