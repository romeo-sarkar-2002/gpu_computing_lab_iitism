#include <stdio.h>
__global__ void helloFromGPU ()
{
    printf ("Hello World from GPU!\n");
    return;
}
void cudaPrintDeviceProp (cudaDeviceProp p)
{
    printf ("name: \"%s\"\n", p.name);
    printf ("totalGlobalMem: %zx\n", p.totalGlobalMem);
    printf ("sharedMemPerBlock: %zx\n", p.sharedMemPerBlock);
    printf ("regsPerBlock: %d\n", p.regsPerBlock);
    printf ("warpSize: %d\n", p.warpSize);
    printf ("memPitch: %zx\n", p.memPitch);
    printf ("maxThreadsPerBlock: %d\n", p.maxThreadsPerBlock);
    printf ("maxThreadsDim: %d:%d:%d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf ("maxGridSize: %d:%d:%d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf ("clockRate: %d\n", p.clockRate);
    printf ("totalConstMem: %zx\n", p.totalConstMem);
    printf ("major: %d\n", p.major);
    printf ("minor: %d\n", p.minor);
    printf ("textureAlignment: %zx\n", p.textureAlignment);
    printf ("texturePitchAlignment: %zx\n", p.texturePitchAlignment);
    printf ("deviceOverlap: %d\n", p.deviceOverlap);
    printf ("multiProcessorCount: %d\n", p.multiProcessorCount);
    printf ("kernelExecTimeoutEnabled: %d\n", p.kernelExecTimeoutEnabled);
    printf ("integrated: %d\n", p.integrated);
    printf ("canMapHostMemory: %d\n", p.canMapHostMemory);
    printf ("computeMode: %d\n", p.computeMode);
    printf ("maxTexture1D: %d\n", p.maxTexture1D);
    printf ("maxTexture1DMipmap: %d\n", p.maxTexture1DMipmap);
    printf ("maxTexture1DLinear: %d\n", p.maxTexture1DLinear);
    printf ("maxTexture2D: %d:%d\n", p.maxTexture2D[0], p.maxTexture2D[1]);
    printf ("maxTexture2DMipmap: %d:%d\n", p.maxTexture2DMipmap[0], p.maxTexture2DMipmap[1]);
    printf ("maxTexture2DLinear: %d:%d:%d\n", p.maxTexture2DLinear[0], p.maxTexture2DLinear[1], p.maxTexture2DLinear[2]);
    printf ("maxTexture2DGather: %d:%d\n", p.maxTexture2DGather[0], p.maxTexture2DGather[1]);
    printf ("maxTexture3D: %d:%d:%d\n", p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2]);
    printf ("maxTexture3DAlt: %d:%d:%d\n", p.maxTexture3DAlt[0], p.maxTexture3DAlt[1], p.maxTexture3DAlt[2]);
    printf ("maxTextureCubemap: %d\n", p.maxTextureCubemap);
    printf ("maxTexture1DLayered: %d:%d\n", p.maxTexture1DLayered[0], p.maxTexture1DLayered[1]);
    printf ("maxTexture2DLayered: %d:%d:%d\n", p.maxTexture2DLayered[0], p.maxTexture2DLayered[1], p.maxTexture2DLayered[2]);
    printf ("maxTextureCubemapLayered: %d:%d\n", p.maxTextureCubemapLayered[0], p.maxTextureCubemapLayered[1]);
    printf ("maxSurface1D: %d\n", p.maxSurface1D);
    printf ("maxSurface2D: %d:%d\n", p.maxSurface2D[0], p.maxSurface2D[1]);
    printf ("maxSurface3D: %d:%d:%d\n", p.maxSurface3D[0], p.maxSurface3D[1], p.maxSurface3D[2]);
    printf ("maxSurface1DLayered: %d:%d\n", p.maxSurface1DLayered[0], p.maxSurface1DLayered[1]);
    printf ("maxSurface2DLayered: %d:%d:%d\n", p.maxSurface2DLayered[0], p.maxSurface2DLayered[1], p.maxSurface2DLayered[2]);
    printf ("maxSurfaceCubemap: %d\n", p.maxSurfaceCubemap);
    printf ("maxSurfaceCubemapLayered: %d:%d\n", p.maxSurfaceCubemapLayered[0], p.maxSurfaceCubemapLayered[1]);
    printf ("surfaceAlignment: %zx\n", p.surfaceAlignment);
    printf ("concurrentKernels: %d\n", p.concurrentKernels);
    printf ("ECCEnabled: %d\n", p.ECCEnabled);
    printf ("pciBusID: %d\n", p.pciBusID);
    printf ("pciDeviceID: %d\n", p.pciDeviceID);
    printf ("pciDomainID: %d\n", p.pciDomainID);
    printf ("tccDriver: %d\n", p.tccDriver);
    printf ("asyncEngineCount: %d\n", p.asyncEngineCount);
    printf ("unifiedAddressing: %d\n", p.unifiedAddressing);
    printf ("memoryClockRate: %d\n", p.memoryClockRate);
    printf ("memoryBusWidth: %d\n", p.memoryBusWidth);
    printf ("l2CacheSize: %d\n", p.l2CacheSize);
    printf ("persistingL2CacheMaxSize: %d\n", p.persistingL2CacheMaxSize);
    printf ("maxThreadsPerMultiProcessor: %d\n", p.maxThreadsPerMultiProcessor);
    printf ("streamPrioritiesSupported: %d\n", p.streamPrioritiesSupported);
    printf ("globalL1CacheSupported: %d\n", p.globalL1CacheSupported);
    printf ("localL1CacheSupported: %d\n", p.localL1CacheSupported);
    printf ("sharedMemPerMultiprocessor: %zx\n", p.sharedMemPerMultiprocessor);
    printf ("regsPerMultiprocessor: %d\n", p.regsPerMultiprocessor);
    printf ("managedMemory: %d\n", p.managedMemory);
    printf ("isMultiGpuBoard: %d\n", p.isMultiGpuBoard);
    printf ("multiGpuBoardGroupID: %d\n", p.multiGpuBoardGroupID);
    printf ("hostNativeAtomicSupported: %d\n", p.hostNativeAtomicSupported);
    printf ("singleToDoublePrecisionPerfRatio: %d\n", p.singleToDoublePrecisionPerfRatio);
    printf ("pageableMemoryAccess: %d\n", p.pageableMemoryAccess);
    printf ("concurrentManagedAccess: %d\n", p.concurrentManagedAccess);
    printf ("computePreemptionSupported: %d\n", p.computePreemptionSupported);
    printf ("canUseHostPointerForRegisteredMem: %d\n", p.canUseHostPointerForRegisteredMem);
    printf ("cooperativeLaunch: %d\n", p.cooperativeLaunch);
    printf ("cooperativeMultiDeviceLaunch: %d\n", p.cooperativeMultiDeviceLaunch);
    printf ("cooperativeMultiDeviceLaunch: %d\n", p.cooperativeMultiDeviceLaunch);
    printf ("sharedMemPerBlockOptin: %zx\n", p.sharedMemPerBlockOptin);
    printf ("pageableMemoryAccessUsesHostPageTables: %d\n", p.pageableMemoryAccessUsesHostPageTables);
    printf ("directManagedMemAccessFromHost: %d\n", p.directManagedMemAccessFromHost);
    printf ("maxBlocksPerMultiProcessor: %d\n", p.maxBlocksPerMultiProcessor);
    printf ("accessPolicyMaxWindowSize: %d\n", p.accessPolicyMaxWindowSize);
    printf ("reservedSharedMemPerBlock: %zx\n", p.reservedSharedMemPerBlock);
    return;
}
int main ()
{
    // printf ("Hello World from CPU!\n");
    // helloFromGPU <<<1, 500>>> ();
    // cudaDeviceReset ();
    int n;
    cudaGetDeviceCount (&n);
    cudaDeviceProp p;
    cudaGetDeviceProperties (&p, 0);
    cudaPrintDeviceProp (p);
    printf ("exit 0");
    
    cudaError_t e = cudaGetLastError ();
    printf ("\"%s\"\n", cudaGetErrorString (e));
    cudaDeviceReset ();
    // printf ("%d\n", n);
    return 0;
}