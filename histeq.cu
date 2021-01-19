#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>

struct color {float r, g, b, a;};


__global__ void gpu_histo_global_atomics( unsigned int* histo_output, cudaTextureObject_t &texObjInput, int W, int H ){
    // histo_output[ahány blokk van *256 érték]
    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;
    //Output index start for this block's histogram:
    int I = B*(256);
    unsigned int* h = histo_output + I;

    // process pixel blocks horizontally
    // updates our block's partial histogram in global memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        float4 pixel = tex2D<float4>(texObjInput, x, y);
        unsigned int intensity = uint((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z) * 255);
        atomicAdd(&h[intensity], 1);
    }
}


/*
__global__ void gpu_histo_shared_atomics( unsigned int* histo_output, cudaTextureObject_t &texObjInput, int W, int H ){
    //munkacsop. nem a globálisban növelget, hanem shared memóriás blokkot: histograms a shared memoryban
    // ha végzett, visszamásoljuk globálba
    __shared__ unsigned int histo[256];

    // texture coordinates:
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x0 - W/2.0f;
    float v = y0 - H/2.0f;

    float4 pixel = tex2D<float4>(texObjInput, u, v);
    unsigned int intensity = uint((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z) * 255);

    //Number of threads in the block:
    int Nthreads = blockDim.x * blockDim.y;
    //Linear thread idx:
    int LinID = threadIdx.x + threadIdx.y * blockDim.x;
    //zero histogram:
    for (int i = LinID; i < 256; i += Nthreads){ histo[i] = 0; }
    __syncthreads();

    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;

    // process pixel blocks horizontally
    // updates the partial histogram in shared memory
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        atomicAdd(&histo[intensity], 1);
    }
    __syncthreads();

    //Output index start for this block's histogram:
    int I = B*256;
    unsigned int* H0 = histo_output + I;

    //Copy shared memory histograms to global memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H0[i] = histo[i];
    }
}
*/

// a munkacsoportokat összegezni kell!
// globális és shared memóriás is ezt használja
__global__ void gpu_histo_accumulate(const unsigned int* in, int nBlocks, unsigned int* out){
    // minden thread egy shade-et
    // minden szál egy csatornát összegez fel a részhistokból: 3*256-os lépéseket tesz. Megismételve minden csatornára és minden rgb értékre a 256 lépcső közül. 3*256 szálat
    // fog futtatni nekünk.
    // szálazonosító:
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 256){
        unsigned int sum = 0;
        for(int j=0; j < nBlocks ; j++){
            sum += in[i + (256) * j];
        }
        out[i] = sum;
    }
}


__global__ void gpu_normalize_image(unsigned int* out, cudaTextureObject_t &texObjInput, const unsigned int* histo, int W, int H){
    // g_ij = floor( (L-1 = 255)* SUM_{n=0}^{f_i,j}[ p_n ])
    // p_n = number of pixels with intensity n / total number of pixels:
    // p_n = histo[n] / (W*H)
    // texture coordinates:
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x0 - W/2.0f;
    float v = y0 - H/2.0f;

    // calc intensity (f_ij):
    float4 pixel = tex2D<float4>(texObjInput, u, v);
    unsigned int fij = uint((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z) * 255);

    unsigned int sum = 0;
    for(int i=0; i<fij; i++){
        sum += histo[i];
    }
    sum = uint(sum * 255.0f/(W*H));

    // g_ij is y*W+x
    out[y0*W+x0] = sum;
}

void debug(const char* txt){
    std::cout << txt << std::endl;
}




int main(void) {
    cudaError_t err = cudaSuccess;
    static const std::string input_filename = "input.png";
    static const std::string output_filename = "texture_out.png";
    static const int block_size = 16;

    /* INPUT TEXTURE */
    int w = 0;
    int h = 0;
    int ch = 0;

    struct rawcolor{unsigned char r, g, b, a;};

    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4));
    if(!data0){
        std::cout << "Could not open input file: " << input_filename << std::endl;
        return 1;
    }
    else{
        std::cout << input_filename << " opened. Width, height, channels: " << w << " " << h << " " << ch << std::endl;
    }


    int nBlocksH = h / block_size;

    std::vector<color> input(w*h);

    std::transform(data0, data0+w*h, input.begin(), 
    [](rawcolor c){ return color{c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f}; }
    );
    stbi_image_free(data0);

    
    
	/* Initialize texture
    Kell egy cudaArray ami az adatot tárolja
    Kell egy cudaResourceDesc ami jellemzi
    Kell cudaTextureDesc ami jellemzi a textúrát
    Creation: cudaCreateTextureObject fv -> cudaTextureObject_t objektum, át lehet adni kernelnek
    */


    // Cuda Array:
    cudaChannelFormatDesc channelDescInput = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
    cudaArray* aInput;

    // Malloc and load to device:
    err = cudaMallocArray(&aInput,&channelDescInput, w, h);
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
    err = cudaMemcpyToArray(aInput, 0, 0, input.data(), w*h*sizeof(color), cudaMemcpyHostToDevice);
    if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

    // cudaResourceDesc írja le, hogy cudaArray-ben van a cucc:
    cudaResourceDesc resdescInput{}; // 0-ra inicializálva
    resdescInput.resType = cudaResourceTypeArray; //megmondja hogy array van
    resdescInput.res.array.array = aInput; // pointer rá

    // cudaTextureDesc jellemzi a textúrát:
    cudaTextureDesc texDesc{}; // 0 init
    texDesc.addressMode[0] = cudaAddressModeClamp; // mi a van a htáron: az utolsó szín
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; //lineáris szűrés
    texDesc.readMode = cudaReadModeElementType; // nincs adatkonv. float-ra
    texDesc.normalizedCoords = 0; // [0.0, texture_size] a tartomány

    // Mindent összekapcs:
    cudaTextureObject_t texObjInput = 0;
    err = cudaCreateTextureObject(&texObjInput, &resdescInput, &texDesc, nullptr); 
    if( err != cudaSuccess){ std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }
        


    /* OUTPUTS */
    // picture Output:
    unsigned int* pOutput = nullptr;
    err = cudaMalloc( (void**)&pOutput, w*h*sizeof(unsigned int) );
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory for pOutput: " << cudaGetErrorString(err) << "\n"; return -1; }

    // részhisztogrammok: :
    unsigned int*  hPartials = nullptr;
    err = cudaMalloc( (void**)&hPartials, nBlocksH*256*sizeof(unsigned int) );
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory for hPartials: " << cudaGetErrorString(err) << "\n"; return -1; }
    err = cudaMemset(hPartials, 0, nBlocksH*256*sizeof(unsigned int) );
    if( err != cudaSuccess){ std::cout << "Error setting memory to zero for hPartials: " << cudaGetErrorString(err) << "\n"; return -1; }

    // input kiszámolt histogramja :
    unsigned int*  hOutput   = nullptr;
    err = cudaMalloc( (void**)&hOutput, 256*sizeof(unsigned int) );
    if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory for hOutput: " << cudaGetErrorString(err) << "\n"; return -1; }

    

    /* RISE KERNEL, RISE */
    /* 1 - calculate partial histos*/
    {
        //dim3 dimGrid( w / block_size, h / block_size );
        //dim3 dimBlock( block_size, block_size );
        dim3 dimGrid( 1, nBlocksH );
        dim3 dimBlock( block_size, block_size );
        gpu_histo_global_atomics<<<dimGrid, dimBlock>>>( hPartials, texObjInput, w, h);
        err = cudaGetLastError();
        if (err != cudaSuccess){ std::cout << "CUDA error in first kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
    }
    debug("Kernel 1 done.");
    // TESTING
    std::vector<unsigned int> hostPartOutput(nBlocksH*256);
    err = cudaMemcpy( hostPartOutput.data(), hPartials, nBlocksH*256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
    if (err != cudaSuccess){ std::cout << "Már ITT (1) baj van: " << cudaGetErrorString(err) << "\n"; return -1; }


    /* 2 - accumulate histos*/
    {
        dim3 dimGrid( 1 );
        dim3 dimBlock( 256 );
        gpu_histo_accumulate<<<dimGrid, dimBlock>>>(hPartials, nBlocksH, hOutput);
        err = cudaGetLastError();
        if (err != cudaSuccess){ std::cout << "CUDA error in second kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
    }
    debug("Kernel 2 done.");

    // TESTING
    std::vector<unsigned int> hostHistOutput(256);
    err = cudaMemcpy( hostHistOutput.data(), hOutput, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost );
    if (err != cudaSuccess){ std::cout << "Már itt (2) baj van: " << cudaGetErrorString(err) << "\n"; return -1; }


    /* 3 - histo equialize*/
    {
        dim3 dimGrid( w / block_size, h / block_size );
        dim3 dimBlock( block_size, block_size );
        gpu_normalize_image<<<dimGrid, dimBlock>>>(pOutput, texObjInput, hOutput, w, h);
        err = cudaGetLastError();
        if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
    }
    debug("Kernel 3 done.");
    cudaDeviceSynchronize();

    // eredmény visszamásolása gazda oldalra
    std::vector<unsigned int> hostOutput(w*h*sizeof(unsigned int));
    err = cudaMemcpy( hostOutput.data(), pOutput, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost );
    if( err != cudaSuccess){ std::cout << "Error copying memory to host at end: " << cudaGetErrorString(err) << "\n"; return -1; }

    /* CONVERT WRITE & FINISH */
    auto convert_and_write = [w, h, ch](std::string const& filename, std::vector<unsigned int> &image ){
        std::vector<color> data(w*h);
        for(int i=0; i< data.size(); i++){
            data[i].r = image[i];
            data[i].g = image[i];
            data[i].b = image[i];
            data[i].a = 255;
        }
        std::vector<rawcolor> tmp(w*h*ch);
        std::transform(data.cbegin(), data.cend(), tmp.begin(),
            [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                            (unsigned char)(c.g*255.0f),
                                            (unsigned char)(c.b*255.0f),
                                            (unsigned char)(c.a*255.0f) }; } );

        int res = stbi_write_jpg(filename.c_str(), w, h, ch, tmp.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    
    convert_and_write(output_filename, hostOutput);
}