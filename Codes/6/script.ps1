nvcc -dc display.cu -o display.obj
nvcc -dc init.cu -o init.obj
nvcc -dc kernel.cu -o kernel.obj
nvcc -dc main.cu -o main.obj
nvcc display.obj init.obj kernel.obj main.obj
rm display.obj
rm init.obj
rm kernel.obj
rm main.obj
if ($?)
{
    .\a.exe
    rm a.exe
    rm a.lib
    rm a.exp
}