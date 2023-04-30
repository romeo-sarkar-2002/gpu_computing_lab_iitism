nvcc -I. -dc matrix.cu -o matrix.obj 
nvcc -I. -dc main.cu -o main.obj 
nvcc matrix.obj main.obj
rm matrix.obj 
rm main.obj
# nvcc main.cu matrix.cu
if ($?)
{
    .\a.exe
    rm a.exe 
    rm a.exp
    rm a.lib
}