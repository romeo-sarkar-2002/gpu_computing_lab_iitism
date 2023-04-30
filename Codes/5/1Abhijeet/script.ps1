nvcc -I. -dc lib.cu -o lib.obj 
nvcc -I. -dc main.cu -o main.obj 
nvcc lib.obj main.obj
rm lib.obj 
rm main.obj
# nvcc main.cu lib.cu
if ($?)
{
    .\a.exe
    rm a.exe 
    rm a.exp
    rm a.lib
}