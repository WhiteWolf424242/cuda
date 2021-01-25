
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
#include <atomic>

struct color {float r, g, b, a;};
struct rawcolor{ unsigned char r, g, b, a; };

__global__ void TEX_identity_trafo(unsigned int* out, cudaTextureObject_t texObjInput, const unsigned int* histo, int W, int H){
    unsigned int x0 = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y0 = blockIdx.y*blockDim.y + threadIdx.y;

    // calc intensity (f_ij):
    float4 pixel = tex2D<float4>(texObjInput, x0, y0);
    unsigned int fij = (unsigned int)(((0.3 * pixel.x) + (0.59 * pixel.y) + (0.11 * pixel.z))*255);

    // most csak identitás trafó (nem használ histo):
    out[y0*W+x0] = fij;
}


int main()
{
    static const std::string input_filename   = "input.jpg";
    static const std::string output_filename1 = "cpu_out1.jpg";
    static const std::string output_filename2 = "gpu_out1.jpg";
    static const std::string output_filename3 = "gpu_out2.jpg";

    static const int block_size = 16;
    //int nBlocksW = 0; //number of blocks horizontally, not used now
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

    cudaError_t err = cudaSuccess;
    std::vector<unsigned int> hostOutput(w*h);
    std::vector<unsigned int> hostOutputShared(w*h);
    unsigned int*  hPartials = nullptr;
    unsigned int*  hOutput   = nullptr;
    unsigned int*  pImageOut = nullptr;

    float dt2 = 0.0f;
    {
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
        texDesc.filterMode = cudaFilterModeLinear; //lineáris szűrés
        texDesc.readMode = cudaReadModeElementType; // nincs adatkonv. float-ra
        texDesc.normalizedCoords = 0; // [0.0, texture_size] a tartomány

        // Mindent összekapcs:
        cudaTextureObject_t texObjInput = 0;
        err = cudaCreateTextureObject(&texObjInput, &resdescInput, &texDesc, nullptr); 
        if( err != cudaSuccess){ std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

        // KERNEL:
        {
            dim3 dimGrid( w / block_size, h / block_size + 1 );
            dim3 dimBlock( block_size, block_size );
            TEX_identity_trafo<<<dimGrid, dimBlock>>>(pImageOut, texObjInput, hOutput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess){ std::cout << "CUDA error in third kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
        }
        cudaDeviceSynchronize();

        // eredmény visszamásolása gazda oldalra
        err = cudaMemcpy( hostOutputShared.data(), pImageOut, w*h*sizeof(unsigned int), cudaMemcpyDeviceToHost );
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

        int res = stbi_write_jpg(filename.c_str(), w, h, 4, tmp.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    convert_and_write(output_filename3, hostOutputShared);
}

