// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <iostream>
#include "DeviceInfo.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * CUDA Kernels
 *
 */



 // 2D Convolution kernel "naive" global memory
__global__ void conv_2D_allglobal(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size)
{
    int columns = blockDim.x * blockIdx.x + threadIdx.x;
    int rows = blockDim.y * blockIdx.y + threadIdx.y;

    if (columns < num_col && rows < num_row) //checking to ensure that we are in bounds within our image
    {
        float result = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) //+/- around center point
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = rows + filter_row;  
                int image_col = columns + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f; //checks to ensure valid image range and row
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2]; //actual filter value
                result += image_value * filter_value;
            }
        }
        d_output[rows * num_col + columns] = result;
    }

}

__global__ void conv_parent(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        conv_2D_allglobal << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, d_filter, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}


__global__ void basic_laplacian(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size)
{
    int columns = blockDim.x * blockIdx.x + threadIdx.x;
    int rows = blockDim.y * blockIdx.y + threadIdx.y;

    if (columns < num_col && rows < num_row) //checking to ensure that we are in bounds within our image
    {
        float result = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) //+/- around center point
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = rows + filter_row;
                int image_col = columns + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row&& image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f; //checks to ensure valid image range and row
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2]; //actual filter value
                result += image_value * filter_value;
            }
        }
        if (result > 255) result = 255;
        else if (result < 0) result = 0;
        d_output[rows * num_col + columns] = result;
    }

}


__global__ void composite_laplacian(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size)
{
    int columns = blockDim.x * blockIdx.x + threadIdx.x;
    int rows = blockDim.y * blockIdx.y + threadIdx.y;

    if (columns < num_col && rows < num_row) //checking to ensure that we are in bounds within our image
    {
        float result = 0.f;

        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) //+/- around center point
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = rows + filter_row;
                int image_col = columns + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row&& image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f; //checks to ensure valid image range and row
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2]; //actual filter value
                result += image_value * filter_value;
            }
        }
        if (result > 255) result = 255;
        else if (result < 0) result = 0;
        d_output[rows * num_col + columns] = result;
    }
}

__global__ void laplace_parent(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        composite_laplacian << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, d_filter, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}


__global__ void sobel(float* d_output, float* d_input, float* d_filter, float* d_filt_transpose, int num_row, int num_col, int filter_size)
{
    int columns = blockDim.x * blockIdx.x + threadIdx.x;
    int rows = blockDim.y * blockIdx.y + threadIdx.y;
    //printf("sobel\n");

    //pick 4 random location on the image, put breakpoint on the host, right before passing to the device, look at the device
    //use the immediate window
    //once I go to the device, put a breakpoint in the kernel then look at the values d_input[some value] -> the values should be the same
    //walk through the loop and look at filter value and look at image value and its corresponding result
    //challenge: when debugging, same line will be hit a lot.  so set a conditional break point.
    //conditional breakpoint -> right click on break point
    if (columns < num_col && rows < num_row)
    {
        float resultx = 0.f;
        float resulty = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = rows + filter_row;
                int image_col = columns + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f; //checks to ensure valid image range and row
                float Gx_filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2]; //actual filter value
                resultx += Gx_filter_value * image_value;

                float Gy_filter_value = d_filt_transpose[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];
                resulty += Gy_filter_value * image_value;
            }
        }
        d_output[rows * num_col + columns] = sqrt(resultx * resultx + resulty*resulty);

    }

}
__global__ void sobel_parent(float* d_output, float* d_input, float* sobel_filter, float* sobel_filt_transpose, int num_row, int num_col, int filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        sobel << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, sobel_filter, sobel_filt_transpose, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}

__global__ void sobel_laplace(float* d_output, float* d_input, float* sobel_filter, float* sobel_filt_transpose, float* laplacian_filter, int num_row, int num_col, int filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{  

    if (threadIdx.x == 0){
        sobel << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, sobel_filter, sobel_filt_transpose, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    basic_laplacian << < blocksPerGrid, threadsPerBlock >> > (d_output, d_output, laplacian_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}

__global__ void sobel_gaussian(float* d_output, float* d_input, float* sobel_filter, float* sobel_filt_transpose, float* gaussian_filter, int num_row, int num_col, int sobel_filter_size, int gaussian_filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        sobel << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, sobel_filter, sobel_filt_transpose, num_row, num_col, sobel_filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    conv_2D_allglobal << < blocksPerGrid, threadsPerBlock >> > (d_output, d_output, gaussian_filter, num_row, num_col, gaussian_filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}

__global__ void laplace_gaussian(float* d_output, float* d_input, float* laplacian_filter, float* gaussian_filter, int num_row, int num_col,int laplace_filter_size, int gaussian_filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        basic_laplacian << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, laplacian_filter, num_row, num_col, laplace_filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();
    conv_2D_allglobal << < blocksPerGrid, threadsPerBlock >> > (d_output, d_output, gaussian_filter, num_row, num_col, gaussian_filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}


__global__ void sobel_laplace_gaussian(float* d_output, float* d_input, float* sobel_filter, float* sobel_filt_transpose, float* laplace_filter, float* gaussian_filter, int num_row, int num_col, int filter_size, int gaussian_filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    sobel << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, sobel_filter, sobel_filt_transpose, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    basic_laplacian << < blocksPerGrid, threadsPerBlock >> > (d_output, d_output, laplace_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    conv_2D_allglobal << < blocksPerGrid, threadsPerBlock >> > (d_output, d_output, gaussian_filter, num_row, num_col, gaussian_filter_size);
    cudaDeviceSynchronize();
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}


__global__ void contrast_enhancement(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size)
{
    int columns = blockDim.x * blockIdx.x + threadIdx.x;
    int rows = blockDim.y * blockIdx.y + threadIdx.y;

    if (columns < num_col && rows < num_row) //checking to ensure that we are in bounds within our image
    {
        float result = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) //+/- around center point
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = rows + filter_row;
                int image_col = columns + filter_col;
                float image_value = (image_row >= 0 && image_row < num_row&& image_col >= 0 && image_col < num_col) ? d_input[image_row * num_col + image_col] : 0.f; //checks to ensure valid image range and row
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2]; //actual filter value
                result += image_value * filter_value;
            }
        }
        if (result > 255) result = 255;
        else if (result < 0) result = 0;
        d_output[rows * num_col + columns] = result;
    }

}

__global__ void contrast_parent(float* d_output, float* d_input, float* d_filter, int num_row, int num_col, int filter_size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    if (threadIdx.x == 0) {
        contrast_enhancement << < blocksPerGrid, threadsPerBlock >> > (d_output, d_input, d_filter, num_row, num_col, filter_size);
        cudaDeviceSynchronize();
    }
    __syncthreads();

    if (cudaSuccess != cudaGetLastError()) { return; }
    if (cudaSuccess != cudaDeviceSynchronize()) { return; }

}

using namespace std;


float gaussBlurFilter_7x7[49] = {
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,	  0.1239f / 3.0f,    0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0386f / 3.0f,    0.0887f / 3.0f,    0.1463f / 3.0f,    0.1729f / 3.0f,    0.1463f / 3.0f,    0.0887f / 3.0f,    0.0386f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,    0.1239f / 3.0f,	   0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f
};

float gaussBlurFilter_5x5[25] = {
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f
};

float compositeLaplacianFilter[9] = {
    -1.0f, -1.0f, -1.0f,
    -1.0f, 9.0f, -1.0f,
    -1.0f, -1.0f, -1.0f
};

float basicLaplacianFilterDiags[9] = {
    1.0f, 1.0f, 1.0f,
    1.0f, -8.0f, 1.0f,
    1.0f, 1.0f, 1.0f
};

float sobelEdgeX[9] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f
};

float sobelEdgeXTranspose[9] = {
    1.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f,
    -1.0f, -2.0f, -1.0f
};


float contrastEnhancementFilter[9] = {
    (0.0f), (-1.0f),(0.0f),
    (-1.0f), (4.5f), (-1.0f),
    (0.0f), (-1.0f), (0.0f)
};
//int filterWidth = 7; //gaussian blur
//int filterWidth = 3; //sobel


/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	int imgRows;
	int imgCols;
	int imgChannels;

    string strOutpath = "C:\\Users\\ajlnm\\Desktop\\Portfolio\\gpu_demo\\";
    string inFile = "Big-Gray-Lena_8bit.png";

    unsigned char* h_inimg_char = stbi_load( (strOutpath + inFile).c_str(), &imgCols, &imgRows, &imgChannels, 0);

    int numElements = imgCols * imgRows;
    size_t size = numElements * sizeof(float);
    printf("[Convolution Filter of [%d x %d] image with filter]\n", imgRows, imgCols);

    float* h_inimg = (float*)malloc(size);
    for (int i = 0; i < numElements; i++)
    {
        h_inimg[i] = (int)h_inimg_char[i]; // convert from char to float, costly...
    }

    float* h_outimg = (float*)malloc(size);
    unsigned char* h_outimg_char = (unsigned char*)malloc(imgRows * imgCols * imgChannels * sizeof(unsigned char));
 
    // Verify that allocations succeeded
    if (h_inimg == NULL || h_outimg == NULL )
    {
        fprintf(stderr, "Failed to allocate host images!\n");
        exit(EXIT_FAILURE);
    }

	// Allocate memory for device input image, filter, output image
	// and make sure it is initialized with the correct data
    float* d_outimg = NULL;
    float* d_inimg = NULL;

    //read lena convert to float, and convert it to unsigned char then write it out and see if I get the same image (do without gpu processing)
	cudaenv::Devices devices(imgRows, imgCols, imgChannels);
	devices.checkCuda((CUresult)cuInit(0));

	devices.checkCuda((CUresult)cudaMalloc((void**)&d_outimg, size)); 
	devices.checkCuda((CUresult)cudaMalloc((void**)&d_inimg, size));

    cout << "Enter 0 for sobel filter" << endl;
    cout << "Enter 1 for basic Laplacian filter" << endl;
    cout << "Enter 2 for Gaussian blur filter" << endl;
    cout << "Enter 3 for sobel + laplacian filter" << endl;
    cout << "Enter 4 for sobel + gaussian filter" << endl;
    cout << "Enter 5 for laplacian + gaussian filter" << endl;
    cout << "Enter 6 for sobel + laplacian + gaussian filter" << endl;
    cout << "Enter 7 for contrast enhancement filter" << endl;

    int choice;

    cin >> choice;

    if (choice == 0)
    {
        printf("you selected sobel filter\n");
        float* d_filt = NULL;
        float* d_filt_transpose = NULL;

        int filterWidth = 3; //sobel
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_filt, sizeof(sobelEdgeX)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_filt_transpose, sizeof(sobelEdgeX)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_filt, sobelEdgeX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_filt_transpose, sobelEdgeXTranspose, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        // Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        sobel_parent << < 1, 1 >> > (d_outimg, d_inimg, d_filt, d_filt_transpose, imgRows, imgCols, filterWidth,threadsPerBlock, blocksPerGrid);
        cudaFree(d_filt);
        cudaFree(d_filt_transpose);
    }

    if (choice == 1)
    {
        printf("you selected laplacian filter\n");
        float* d_filt = NULL;
        int filterWidth = 3;
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_filt, sizeof(basicLaplacianFilterDiags)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_filt, basicLaplacianFilterDiags, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        // Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        laplace_parent << < 1, 1 >> > (d_outimg, d_inimg, d_filt, imgRows, imgCols, filterWidth,threadsPerBlock, blocksPerGrid );
        cudaFree(d_filt);
    }


    if (choice == 2)
    {
        printf("you selected gaussian filter\n");
        float* d_filt = NULL;
        int filterWidth = 7; //gaussian blur
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_filt, sizeof(gaussBlurFilter_7x7)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_filt, gaussBlurFilter_7x7, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        //Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        conv_parent <<< blocksPerGrid, threadsPerBlock >>> (d_outimg, d_inimg, d_filt, imgRows, imgCols, filterWidth, threadsPerBlock, blocksPerGrid);
        cudaFree(d_filt);
    }

    if (choice == 3)
    {
        printf("you selected sobel filter and laplace filter\n");
        float* d_sobel = NULL;
        float* d_sobelTranspose = NULL;
        float* d_laplacian = NULL;
        int filterWidth = 3; 
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobel, sizeof(sobelEdgeX)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobelTranspose, sizeof(sobelEdgeXTranspose)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_laplacian, sizeof(basicLaplacianFilterDiags)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobel, sobelEdgeX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobelTranspose, sobelEdgeXTranspose, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_laplacian, basicLaplacianFilterDiags, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        //Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        sobel_laplace << < 1, 1 >> > (d_outimg, d_inimg, d_sobel, d_sobelTranspose, d_laplacian, imgRows, imgCols, filterWidth,threadsPerBlock, blocksPerGrid);
        cudaFree(d_sobel);
        cudaFree(d_sobelTranspose);
        cudaFree(d_laplacian);
    }

    if (choice == 4)
    {
        printf("you selected sobel filter and gaussian filter\n");
        float* d_sobel = NULL;
        float* d_sobelTranspose = NULL;
        float* d_gaussian = NULL;
        int sobel_filterWidth = 3; 
        int gaussian_filterWidth = 7;
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobel, sizeof(sobelEdgeX)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobelTranspose, sizeof(sobelEdgeXTranspose)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_gaussian, sizeof(gaussBlurFilter_7x7)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobel, sobelEdgeX, sobel_filterWidth * sobel_filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobelTranspose, sobelEdgeXTranspose, sobel_filterWidth * sobel_filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_gaussian, gaussBlurFilter_7x7, gaussian_filterWidth * gaussian_filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        //Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        sobel_gaussian << < 1, 1 >> > (d_outimg, d_inimg, d_sobel, d_sobelTranspose, d_gaussian, imgRows, imgCols, sobel_filterWidth,gaussian_filterWidth, threadsPerBlock, blocksPerGrid);
        cudaFree(d_sobel);
        cudaFree(d_sobelTranspose);
        cudaFree(d_gaussian);
    }

    if (choice == 5)
    {
        printf("you selected laplace filter and gaussian filter\n");
        float* d_laplacian = NULL;
        float* d_gaussian = NULL;
        int gaussian_filterWidth = 7; //gaussian blur
        int laplacian_filterWidth = 3;
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_laplacian, sizeof(basicLaplacianFilterDiags)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_gaussian, sizeof(gaussBlurFilter_7x7)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_laplacian, basicLaplacianFilterDiags, laplacian_filterWidth* laplacian_filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_gaussian, gaussBlurFilter_7x7, gaussian_filterWidth* gaussian_filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        //Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        laplace_gaussian << < 1, 1 >> > (d_outimg, d_inimg, d_laplacian, d_gaussian, imgRows, imgCols, laplacian_filterWidth, gaussian_filterWidth, threadsPerBlock, blocksPerGrid);
        cudaFree(d_laplacian);
        cudaFree(d_gaussian);
    }

    if (choice == 6)
    {
        printf("you selected sobel filter, laplace filter, and gaussian filter\n");
        float* d_sobel = NULL;
        float* d_sobelTranspose = NULL;
        float* d_laplacian = NULL;
        float* d_gaussian = NULL;
        int filterWidth = 3; //gaussian blur
        int gaussian_filterWidth = 7;
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobel, sizeof(sobelEdgeX)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_sobelTranspose, sizeof(sobelEdgeXTranspose)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_laplacian, sizeof(basicLaplacianFilterDiags)));
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_gaussian, sizeof(gaussBlurFilter_7x7)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobel, sobelEdgeX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_sobelTranspose, sobelEdgeXTranspose, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_laplacian, basicLaplacianFilterDiags, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_gaussian, gaussBlurFilter_7x7, gaussian_filterWidth* gaussian_filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        //Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        sobel_laplace_gaussian << < 1, 1 >> > (d_outimg, d_inimg, d_sobel, d_sobelTranspose, d_laplacian, d_gaussian, imgRows, imgCols, filterWidth, gaussian_filterWidth, threadsPerBlock, blocksPerGrid);
        cudaFree(d_sobel);
        cudaFree(d_sobelTranspose);
        cudaFree(d_laplacian);
    }

    if (choice == 7)
    {
        printf("contrast enhancement filter\n");
        float* d_filt = NULL;
        int filterWidth = 3;
        devices.checkCuda((CUresult)cudaMalloc((void**)&d_filt, sizeof(contrastEnhancementFilter)));
        devices.checkCuda((CUresult)cudaMemcpy(d_inimg, h_inimg, size, cudaMemcpyHostToDevice));
        devices.checkCuda((CUresult)cudaMemcpy(d_filt, contrastEnhancementFilter, filterWidth* filterWidth * sizeof(float), cudaMemcpyHostToDevice));
        // Launch the CUDA Kernel
        dim3 threadsPerBlock{ 16,16,1 };
        dim3 blocksPerGrid{ (imgCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgRows + threadsPerBlock.y - 1) / threadsPerBlock.y, 1 };
        printf("CUDA kernel launch exec cfg with (%d,%d,%d) blocks of (%d,%d,%d) threads\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        contrast_parent << < 1, 1 >> > (d_outimg, d_inimg, d_filt, imgRows, imgCols, filterWidth, threadsPerBlock, blocksPerGrid);
        cudaFree(d_filt);
    }



    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

	// retrieve result image from device
	devices.checkCuda((CUresult)cudaMemcpy(h_outimg, d_outimg, size, cudaMemcpyDeviceToHost));

    // convert float to char (lossy...)
    for (int i = 0; i < numElements; i++)
    {
        h_outimg_char[i] = (unsigned char)(unsigned int)h_outimg[i]; // convert from float to 8-bit char, costly...
    }


    string strOutfile = "result_image.png";
    int stbErr = stbi_write_jpg((strOutpath + strOutfile).c_str(), imgCols, imgRows, imgChannels, h_outimg_char, 100);

    // Free device global memory


    // Free host memory
    stbi_image_free(h_inimg);
    free(h_outimg);
	cudaFree(d_outimg);
	cudaFree(d_inimg);

    printf("Done\n");
    return 0;
}
