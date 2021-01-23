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

struct color {unsigned char r, g, b, a;};

using namespace std;

int main()
{
    static const std::string input_filename   = "input.jpg";
    static const std::string output_filename2 = "gpu_out1.jpg";

    
    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components

    // Beolvas
    color* data0 = reinterpret_cast<color*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }
    
    // Convert -> Greyscale
    std::vector<unsigned int> test(w*h);
    for(int i=0; i< test.size(); i++){
        test[i] = (unsigned int)(((0.3 * data0[i].r) + (0.59 * data0[i].g) + (0.11 * data0[i].b)));
    }


    // Write
    struct rawcolor{ unsigned char r, g, b, a; };
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
            [](color c){ return rawcolor{   (unsigned char)((c.r)*255.0f),
                                            (unsigned char)((c.g)*255.0f),
                                            (unsigned char)((c.b)*255.0f),
                                            (unsigned char)((c.a)*255.0f) }; } );

        int res = stbi_write_jpg(filename.c_str(), w, h, ch, tmp.data(), 40);
        if(res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }else{ std::cout << "Output written to file " << filename << "\n"; }
    };

    convert_and_write(output_filename2, test);
}