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
#include <array>
#include <thread>

struct color {float r, g, b, a;};               // for texture
struct rawcolor{ unsigned char r, g, b, a; };   // for buffers


// CPU VÁLTOZAT //
void CPU_normalize_image(std::vector<unsigned int> &output, rawcolor* input, int W, int H){
    std::vector<unsigned int> histo(256,0);

    rawcolor pixels;
    for(int i=0; i< W*H ; i++){
        pixels = input[i];
        int fij = (unsigned int)((0.3 * pixels.r) + (0.59 * pixels.g) + (0.11 * pixels.b));
        histo[fij] += 1;
    }

    for(int i=0 ; i<W*H; i++){
        unsigned int sum = 0;
        pixels = input[i];
        int fij = (unsigned int)((0.3 * pixels.r) + (0.59 * pixels.g) + (0.11 * pixels.b));

        for(int j=0; j<=fij; j++){
            sum += histo[j];
        }
        sum = (unsigned int)(sum * 255.0f/(W*H));
        output[i] = sum;
    }
}


// BUFFERES VÁLTOZAT W/ GLOBAL & SHARED ATOMICS //
// global
__global__ void BUFF_gpu_histo_global_atomics( unsigned int* output, uchar4* input, int W ){
    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;

    //Output index start for this block's histogram:
    int I = B*(256);
    unsigned int* H = output + I;

    // process pixel blocks horizontally
    // updates our block's partial histogram in global memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        uchar4 pixels = input[y * W + x];
        unsigned int fij = (unsigned int)(((0.3 * pixels.x) + (0.59 * pixels.y) + (0.11 * pixels.z)));
        atomicAdd(H+fij, 1);
    }
}
// shared
__global__ void BUFF_gpu_histo_shared_atomics( unsigned int* output, uchar4* input, int W ){
    __shared__ unsigned int histo[256];

    int Nthreads = blockDim.x * blockDim.y;
    int LinID = threadIdx.x + threadIdx.y * blockDim.x;
    //zero histogram:
    for (int i = LinID; i < 256; i += Nthreads){ histo[i] = 0; }
    __syncthreads();

    // linear block index within 2D grid
    int B = blockIdx.x + blockIdx.y * gridDim.x;

    // process pixel blocks horizontally
    // updates the partial histogram in shared memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        uchar4 pixels = input[y * W + x];
        unsigned int fij = (unsigned int)(((0.3 * pixels.x) + (0.59 * pixels.y) + (0.11 * pixels.z)));
        atomicAdd(histo+fij, 1);
    }
    __syncthreads();

    //Output index start for this block's histogram:
    int I = B*(256);
    unsigned int* H = output + I;

    //Copy shared memory histograms to global memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H[i] = histo[i];
    }


}


__global__ void BUFF_gpu_histo_accumulate(const unsigned int* in, int nBlocks, unsigned int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 256)
    {
        unsigned int sum = 0;
        for(int j = 0; j < nBlocks; j++)
        {
            sum += in[i + (256) * j];
        }            
        out[i] = sum;
    }
}

__global__ void BUFF_gpu_normalize_image(unsigned int* out, uchar4* input, const unsigned int* histo, int W, int H){
    // g_ij = floor( (L-1 = 255)* SUM_{n=0}^{f_i,j}[ p_n ])
    // p_n = number of pixels with intensity n / total number of pixels:
    // p_n = histo[n] / (W*H)

    // linear block index within 2D grid (not used)
    //int B = blockIdx.x + blockIdx.y * gridDim.x;

    // process pixel blocks horizontally
    // updates our block's partial histogram in global memory
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int x = threadIdx.x; x < W; x += blockDim.x)
    {
        uchar4 pixel = input[y * W + x];
        int fij = (unsigned int)(((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z)));

        unsigned int sum = 0;
        for(int i=0; i<=fij; i++){
            sum += histo[i];
        }
        sum = (unsigned int)(sum * 255.0f/(W*H));
    
        // g_ij is y*W+x
        out[y * W + x] = sum;
    }
}


// TEXTÚRÁS VÁLTOZAT W/ GLOBAL ATOMICS //

__global__ void TEX_gpu_histo_global_atomics( unsigned int* output, cudaTextureObject_t texObjInput, int W, int H0 )
{
    // texture coordinates:
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    float4 pixels = tex2D<float4>(texObjInput, x0+0.5, y0+0.5);
    unsigned int fij = (unsigned int)(((0.3 * pixels.x) + (0.59 * pixels.y) + (0.11 * pixels.z))*255);

    atomicAdd(output+fij,1);
}

/* // TEX-SHARED ATOMICS lenne - nem tudtam kitalálni.
__global__ void TEX_gpu_histo_shared_atomics( unsigned int* histo_output, cudaTextureObject_t texObjInput, int W, int H ){
    //munkacsop. nem a globálisban növelget, hanem shared memóriás blokkot: histograms a shared memoryban
    // ha végzett, visszamásoljuk globálba
    __shared__ unsigned int histo[256];

    // texture coordinates:
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    float4 pixel = tex2D<float4>(texObjInput, x0+0.5, y0+0.5);
    unsigned int intensity = (unsigned int)(((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z))*255);

    //Number of threads in the block:
    int Nthreads = blockDim.x * blockDim.y;
    //Linear thread idx:
    int LinID = threadIdx.x + threadIdx.y * blockDim.x;
    //zero histogram:
    for (int i = LinID; i < 256; i += Nthreads){ histo[i] = 0; }
    __syncthreads();

    atomicAdd(&histo[intensity], 1);
    __syncthreads();


    int B = blockIdx.x + blockIdx.y * gridDim.x;
    int I = B*256;
    unsigned int* H0 = histo_output + I;

    //Copy shared memory histograms to global memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
       //atomicAdd(histo_output+i,histo[i]);
       H0[i] = histo[i];
    }
}
__global__ void TEX_gpu_histo_accumulate(const unsigned int* in, int nBlocks, unsigned int* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 256){
        unsigned int sum = 0;
        for(int j=0; j < nBlocks ; j++){
            sum += in[i + (256) * j];
        }
        out[i] = sum;
    }
}
*/

__global__ void TEX_gpu_normalize_image(unsigned int* out, cudaTextureObject_t texObjInput, const unsigned int* histo, int W, int H){
    // g_ij = floor( (L-1 = 255)* SUM_{n=0}^{f_i,j}[ p_n ])
    // p_n = number of pixels with intensity n / total number of pixels:
    // p_n = histo[n] / (W*H)
    // texture coordinates:
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    // calc intensity (f_ij):
    float4 pixel = tex2D<float4>(texObjInput, x0+0.5, y0+0.5);
    unsigned int fij = (unsigned int)(((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z))*255);

    unsigned int sum = 0;
    for(int i=0; i<=fij; i++){
        sum += histo[i];
    }
    sum = (unsigned int)(sum * 255.0f/(W*H));

    // g_ij is y*W+x
    out[y0*W+x0] = sum;
}


int main()
{
    static const std::string input_filename   = "input.jpg";
    static const std::string output_filename1 = "cpu_out.jpg";
    static const std::string output_filename2 = "gpu_out1_buff-gl.jpg";
    static const std::string output_filename3 = "gpu_out2_buff-sh.jpg";
    static const std::string output_filename4 = "gpu_out3_tex-gl.jpg";

    static const int block_size = 16;
    int nBlocksH = 0; //number of blocks vertically
    
    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        //nBlocksW = w / block_size; //not used now
        nBlocksH = h / block_size;
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }
    
    // CPU VERSION //
    std::vector<unsigned int> cOutput(w*h);
    auto t0 = std::chrono::high_resolution_clock::now();
    CPU_normalize_image(cOutput, data0, w, h);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Computation took:                    " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.0f << " ms\n";


    cudaError_t err = cudaSuccess;
    std::vector<unsigned int> hostOutput(w*h);
    std::vector<unsigned int> hostOutputShared(w*h);
    std::vector<unsigned int> hostOutputTEX(w*h);

    unsigned char* pInput    = nullptr;
    unsigned int*  hPartials = nullptr;
    unsigned int*  hOutput   = nullptr;
    unsigned int*  pImageOut = nullptr;

    //GPU version using global atomics:
    float dt1 = 0.0f;
    {

        err = cudaMalloc( (void**)&pInput, w*h*sizeof(rawcolor) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc( (void**)&hPartials, nBlocksH*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(hPartials, 0, nBlocksH*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        err = cudaMalloc( (void**)&hOutput, 256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc( (void**)&pImageOut, w*h*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }
        err = cudaMemset(pImageOut, 0, w*h*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy( pInput, data0, w*h*sizeof(rawcolor), cudaMemcpyHostToDevice );
        if( err != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }


        cudaEvent_t evt[6];
        for(auto& e : evt){ cudaEventCreate(&e); }
        //First kernel of histograms:
        {
            dim3 dimGrid( 1, nBlocksH + 1);
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[0]);
            BUFF_gpu_histo_global_atomics<<<dimGrid, dimBlock>>>(hPartials, (uchar4*)pInput, w);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in first kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }
        //Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 256 );
            cudaEventRecord(evt[2]);
            BUFF_gpu_histo_accumulate<<<dimGrid, dimBlock>>>(hPartials, nBlocksH, hOutput);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in second kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[3]);
        }

        // Third kernel: normalize image
        {
            dim3 dimGrid( 1, nBlocksH + 1);  // +1 to nBlocksH so that last row also gets done (megoldja black csíkot alul)
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[4]);
            BUFF_gpu_normalize_image<<<dimGrid, dimBlock>>>(pImageOut, (uchar4*)pInput, hOutput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[5]);
        }
        cudaDeviceSynchronize();

        //Calculate time:
        cudaEventSynchronize(evt[3]);
        float dt = 0.0f;//milliseconds
        cudaEventElapsedTime(&dt, evt[0], evt[1]);
        dt1 = dt;
        cudaEventElapsedTime(&dt, evt[2], evt[3]);
        dt1 += dt;
        cudaEventElapsedTime(&dt, evt[4], evt[5]);
        dt1 += dt;
        for(auto& e : evt){ cudaEventDestroy(e); }
        std::cout << "GPU-BUF global atomics computation took: " << dt1  << " ms\n";

        // eredmény visszamásolása gazda oldalra
        err = cudaMemcpy( hostOutput.data(), pImageOut, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host at end: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    // GPU version still with buffers but shared atomics
    float dt2 = 0.0f;
    {
        // reset outputs
        err = cudaMemset(hPartials, 0, nBlocksH*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }
        
        err = cudaMemset(hOutput, 0, 256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(pImageOut, 0, w*h*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEvent_t evt[6];
        for(auto& e : evt){ cudaEventCreate(&e); }
        //First kernel of global histograms:
        {
            dim3 dimGrid( 1, nBlocksH + 1);
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[0]);
            BUFF_gpu_histo_shared_atomics<<<dimGrid, dimBlock>>>(hPartials, (uchar4*)pInput, w);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in first kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }
        //Second kernel: accumulate partial results:
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 256 );
            cudaEventRecord(evt[2]);
            BUFF_gpu_histo_accumulate<<<dimGrid, dimBlock>>>(hPartials, nBlocksH, hOutput);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in second kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[3]);
        }

        // Third kernel: normalize image
        {
            dim3 dimGrid( 1, nBlocksH + 1);
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[4]);
            BUFF_gpu_normalize_image<<<dimGrid, dimBlock>>>(pImageOut, (uchar4*)pInput, hOutput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[5]);
        }
        cudaDeviceSynchronize();

        //Calculate time:
        cudaEventSynchronize(evt[3]);
        float dt = 0.0f;//milliseconds
        cudaEventElapsedTime(&dt, evt[0], evt[1]);
        dt2 = dt;
        cudaEventElapsedTime(&dt, evt[2], evt[3]);
        dt2 += dt;
        cudaEventElapsedTime(&dt, evt[4], evt[5]);
        dt2 += dt;
        for(auto& e : evt){ cudaEventDestroy(e); }
        std::cout << "GPU-BUF shared atomics computation took: " << dt2  << " ms\n";

        // eredmény visszamásolása gazda oldalra
        err = cudaMemcpy( hostOutputShared.data(), pImageOut, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host at end: " << cudaGetErrorString(err) << "\n"; return -1; }
    }

    err = cudaFree( pInput );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }



    // GPU VERSION WITH TEXTURES - GLOBAL ATOMICS

    float dt3 = 0.0f;
    {
        // reset outputs
        err = cudaMemset(hPartials, 0, nBlocksH*256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(hOutput, 0, 256*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemset(pImageOut, 0, w*h*sizeof(unsigned int) );
        if( err != cudaSuccess){ std::cout << "Error setting memory to zero: " << cudaGetErrorString(err) << "\n"; return -1; }


        // Initialize texture
        // Ehhez transform data to color:
        std::vector<color> input(w*h);
        std::transform(data0, data0+w*h, input.begin(), 
        [](rawcolor c){ return color{c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f}; }
        );

        //Kell egy cudaArray ami az adatot tárolja
        //Kell egy cudaResourceDesc ami jellemzi
        //Kell cudaTextureDesc ami jellemzi a textúrát
        //Creation: cudaCreateTextureObject fv -> cudaTextureObject_t objektum, át lehet adni kernelnek
        
        // Cuda Array:
        cudaChannelFormatDesc channelDescInput = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
        cudaArray* aInput;

        // Malloc and load to device:
        err = cudaMallocArray(&aInput,&channelDescInput, w, h);
        if( err != cudaSuccess){ std::cout << "Error allocating CUDA memory (tex): " << cudaGetErrorString(err) << "\n"; return -1; }
        err = cudaMemcpyToArray(aInput, 0, 0, input.data(), w*h*sizeof(color), cudaMemcpyHostToDevice);
        if( err != cudaSuccess){ std::cout << "Error copying memory to device (tex): " << cudaGetErrorString(err) << "\n"; return -1; }

        // cudaResourceDesc írja le, hogy cudaArray-ben van a cucc:
        cudaResourceDesc resdescInput{}; // 0-ra inicializálva
        resdescInput.resType = cudaResourceTypeArray; //megmondja hogy array van
        resdescInput.res.array.array = aInput; // pointer rá

        // cudaTextureDesc jellemzi a textúrát:
        cudaTextureDesc texDesc{}; // 0 init
        texDesc.addressMode[0] = cudaAddressModeClamp; // mi a van a htáron: az utolsó szín: Clamp
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear; //lineáris szűrés -> koordinátákhoz +0.5 kiolvasásnál hogy ne az interpolációt olvassa
        texDesc.readMode = cudaReadModeElementType; // nincs adatkonv. float-ra
        texDesc.normalizedCoords = 0; // [0.0, texture_size] a tartomány

        // Mindent összekapcs:
        cudaTextureObject_t texObjInput = 0;
        err = cudaCreateTextureObject(&texObjInput, &resdescInput, &texDesc, nullptr); 
        if( err != cudaSuccess){ std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEvent_t evt[6];
        for(auto& e : evt){ cudaEventCreate(&e); }
        // 1 - calculate full histos ( in 1 step)//
        {
            dim3 dimGrid( w / block_size, h / block_size + 1 );
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[0]);
            //TEX_gpu_histo_shared_atomics<<<dimGrid, dimBlock>>>( hPartials, texObjInput, w, h); - nem tudtam működésre bírni.
            TEX_gpu_histo_global_atomics<<<dimGrid, dimBlock>>>( hOutput, texObjInput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in first kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        /* NOT USED by TEX_gpu_histo_global_atomics
        // 2 - accumulate histos//
        {
            dim3 dimGrid( 1 );
            dim3 dimBlock( 256 );
            cudaEventRecord(evt[2]);
            TEX_gpu_histo_accumulate<<<dimGrid, dimBlock>>>(hPartials, nBlocksH, hOutput);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in second kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[3]);
        }
        */ 

        // 3- normalize //
        {
            dim3 dimGrid( w / block_size, h / block_size + 1);
            dim3 dimBlock( block_size, block_size );
            cudaEventRecord(evt[4]);
            TEX_gpu_normalize_image<<<dimGrid, dimBlock>>>(pImageOut, texObjInput, hOutput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[5]);
        }
        cudaDeviceSynchronize();
    
        //Calculate time:
        cudaEventSynchronize(evt[5]);
        float dt = 0.0f;//milliseconds
        cudaEventElapsedTime(&dt, evt[0], evt[1]);
        dt3 = dt;
        cudaEventElapsedTime(&dt, evt[2], evt[3]);
        dt3 += dt;
        cudaEventElapsedTime(&dt, evt[4], evt[5]);
        dt3 += dt;
        for(auto& e : evt){ cudaEventDestroy(e); }
        std::cout << "GPU-TEX global atomics computation took: " << dt3  << " ms\n";

        // eredmény visszamásolása gazda oldalra
        err = cudaMemcpy( hostOutputTEX.data(), pImageOut, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost );
        if( err != cudaSuccess){ std::cout << "Error copying memory to host at end: " << cudaGetErrorString(err) << "\n"; return -1; }
    }


    stbi_image_free(data0);
    err = cudaFree( hPartials );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( hOutput );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

    err = cudaFree( pImageOut );
    if( err != cudaSuccess){ std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }





    /* CONVERT WRITE & FINISH */
    auto convert_and_write = [w, h](std::string const& filename, std::vector<unsigned int> &image ){
        std::vector<rawcolor> data(w*h);
        for(int i=0; i< data.size(); i++){
            data[i].r = image[i];
            data[i].g = image[i];
            data[i].b = image[i];
            data[i].a = 255;
        }
        std::vector<rawcolor> tmp(w*h);
        std::transform(data.cbegin(), data.cend(), tmp.begin(),
            [](rawcolor c){ return rawcolor{   (unsigned char)(c.r),
                                            (unsigned char)(c.g),
                                            (unsigned char)(c.b),
                                            (unsigned char)(c.a) }; } );

        int res = stbi_write_jpg(filename.c_str(), w, h, 4, tmp.data(), 100);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    convert_and_write(output_filename1, cOutput);
    convert_and_write(output_filename2, hostOutput);
    convert_and_write(output_filename3, hostOutputShared);
    convert_and_write(output_filename4, hostOutputTEX);

}
