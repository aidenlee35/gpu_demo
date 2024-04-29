#ifndef DEVICEINFO_H
#define DEVICEINFO_H

#include <cuda.h>
#include <stdio.h>
#include <string>
#include <assert.h>
#include <device_launch_parameters.h>
#include "read_source.h"
#include <iostream>
#include <cstring>


namespace cudaenv
{
	class Devices
	{
	public:
		Devices() = default;
		Devices(int Row, int Col)
		{
			numRow = Row;
			numCol = Col;
		}
		Devices(int Row, int Col, int Channel)
		{
			numRow = Row;
			numCol = Col;
			numChannel = Channel;
		}
		Devices(const Devices& other) = default;
		Devices(Devices && other) = default;
		~Devices() = default;
		CUresult checkCuda(CUresult result)
		{
			if (result != CUDA_SUCCESS) {
				const char* message;
				fprintf(stderr, "CUDA Runtime Error: %s\n", cuGetErrorString(result, &message));
				assert(result == CUDA_SUCCESS);
			}
			return result;
		}

//		CUresult getModuleFromPTX(CUmodule* module)
//		{
//			nvrtcResult nvrtcRes;
//			CUresult cuRes;
//			std::string cuFilename = "C:\\Users\\ajlnm\\Desktop\\High_performance_GPU_computing\\week1\\Homework1_re\\Homework1\\Cuda\\kernel.cuh";
//			//std::string cuFilename = "C:\\Users\\PhotonUser\\My Files\\Temporary Files\\CudaRuntime1\\CudaRuntime1\\kernel.cuh";
//			size_t file_size;
//			char* kernel_source = read_source(cuFilename.c_str(), &file_size);
//
//			if (NULL == kernel_source)
//			{
//				printf("Error: Failed to read kernel source code from file name: %s!\n", cuFilename.c_str());
//			}
//#ifdef _DEBUG
//			printf("%s\n", &kernel_source[3]);
//#endif
//			nvrtcProgram prog;
//			nvrtcRes = nvrtcCreateProgram(&prog,         // prog
//				&kernel_source[3],         // buffer
//				cuFilename.c_str(),    // name
//				0,             // numHeaders
//				NULL,          // headers
//				NULL);         // includeNames
//
//
//#ifdef _DEBUG
//			const char* opts[] = { "--gpu-architecture=compute_75","--fmad=false","--device-debug" };
//			nvrtcRes = nvrtcCompileProgram(prog,     // prog
//				3,        // numOptions
//				opts);    // options
//
//#else
//			const char* opts[] = { "--gpu-architecture=compute_75","--fmad=false" };
//			nvrtcRes = nvrtcCompileProgram(prog,     // prog
//				2,        // numOptions
//				opts);    // options
//#endif
//
//	// Obtain compilation log from the program.
//			size_t logSize;
//			nvrtcRes = nvrtcGetProgramLogSize(prog, &logSize);
//			char* log = new char[logSize];
//			nvrtcRes = nvrtcGetProgramLog(prog, log);
//			// Obtain PTX from the program.
//			size_t ptxSize;
//			nvrtcRes = nvrtcGetPTXSize(prog, &ptxSize);
//			char* ptx = new char[ptxSize];
//			nvrtcRes = nvrtcGetPTX(prog, ptx);
//			nvrtcRes = nvrtcDestroyProgram(&prog);
//
//			cuRes = cuModuleLoadDataEx(module, ptx, 0, 0, 0);
//
//			free(kernel_source);
//
//			return CUDA_SUCCESS;
//
//		}

		//double data type
		void printMat(double* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%2.2f \t", *(h_a + row * numCol + col));
				}
				printf("\n");
			}
		}

		void linToMat(double* h_c, double h_clin[], int vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					*(h_c + row * numCol + col) = h_clin[vecIndex];
					vecIndex++;
				}
			}
		}

		void MatToLin(double* h_a, double* h_b, double* h_c, double h_alin[], double h_blin[], double h_clin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount] = *(h_a + row * numCol + col);
					h_blin[linCount] = *(h_b + row * numCol + col);
					h_clin[linCount] = *(h_c + row * numCol + col);
					linCount++;
				}
			}
		}

		void initMat(double* h_a, double* h_b, double* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					*(h_a + row * numCol + col) = 1;
					*(h_b + row * numCol + col) = 2;
					*(h_c + row * numCol + col) = 0;

				}
			}

		}

		void initConst(double* h_a, double* h_b)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					*(h_a + row * numCol + col) = 2.6;
					*(h_b + row * numCol + col) = 3.3;
				}
			}
		}

		void constMatToconstLin(double* h_a, double* h_b, double h_alin[], double h_blin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount] = *(h_a + row * numCol + col);
					h_blin[linCount] = *(h_b + row * numCol + col);
					linCount++;
				}
			}
		}

		//int data type
		void printMat(int* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%d \t", *(h_a + row * numCol + col));
				}
				printf("\n");
			}
		}


		void linToMat(int* h_c, int h_clin[], int vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					*(h_c + row * numCol + col) = h_clin[vecIndex];
					vecIndex++;
				}
			}
		}


		void MatToLin(int* h_a, int* h_b, int* h_c, int h_alin[], int h_blin[], int h_clin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount] = *(h_a + row * numCol + col);
					h_blin[linCount] = *(h_b + row * numCol + col);
					h_clin[linCount] = *(h_c + row * numCol + col);
					linCount++;
				}
			}
		}



		void initMat(int* h_a, int* h_b, int* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					*(h_a + row * numCol + col) = 5;
					*(h_b + row * numCol + col) = 2;
					*(h_c + row * numCol + col) = 0;

				}
			}

		}

		//void initMat(int* h_a, int* h_b, int* h_c)
		//{
		//	*(h_a + 0) = 0;
		//	*(h_a + 1) = 1;
		//	*(h_a + 2) = 2;
		//	*(h_a + 3) = 3;
		//	*(h_a + 4) = 4;
		//	*(h_a + 5) = 5;
		//	*(h_a + 6) = 6;
		//	*(h_a + 7) = 7;
		//	*(h_a + 8) = 8;
		//	*(h_a + 9) = 9;
		//	*(h_c + 0) = 0;
		//	*(h_c + 1) = 0;
		//	*(h_c + 2) = 0;
		//	*(h_c + 3) = 0;
		//	*(h_c + 4) = 0;
		//	*(h_c + 5) = 0;
		//	*(h_c + 6) = 0;
		//	*(h_c + 7) = 0;
		//	*(h_c + 8) = 0;
		//	*(h_c + 9) = 0;
		//	//for (int row = 0; row < numRow; row++)
		//	//{
		//	//	for (int col = 0; col < numCol; col++)
		//	//	{
		//	//		//*(h_a + row * numCol + col) = 5;
		//	//		*(h_b + row * numCol + col) = 2;
		//	//		*(h_c + row * numCol + col) = 0;

		//	//	}
		//	//}

		//}

		//unsigned char data type
		void printMat(unsigned char* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%d \t", *(h_a + row * numCol + col));
				}
				printf("\n");
			}
		}


		void linToMat(unsigned char* h_c, unsigned char h_clin[], unsigned char vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					*(h_c + row * numCol + col) = h_clin[vecIndex];
					vecIndex++;
				}
			}
		}


		void MatToLin(unsigned char* h_a, unsigned char* h_b, unsigned char* h_c, unsigned char h_alin[], unsigned char h_blin[], unsigned char h_clin[], unsigned char vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount] = *(h_a + row * numCol + col);
					h_blin[linCount] = *(h_b + row * numCol + col);
					h_clin[linCount] = *(h_c + row * numCol + col);
					linCount++;
				}
			}
		}



		void initMat(unsigned char* h_a, unsigned char* h_b, unsigned char* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					*(h_a + row * numCol + col) = 5;
					*(h_b + row * numCol + col) = 2;
					*(h_c + row * numCol + col) = 0;

				}
			}

		}

		//void initMat(unsigned char* h_a, unsigned char* h_b, unsigned char* h_c)
		//{
		//	*(h_a + 0) = 0;
		//	*(h_a + 1) = 1;
		//	*(h_a + 2) = 2;
		//	*(h_a + 3) = 3;
		//	*(h_a + 4) = 4;
		//	*(h_a + 5) = 5;
		//	*(h_a + 6) = 6;
		//	*(h_a + 7) = 7;
		//	*(h_a + 8) = 8;
		//	*(h_a + 9) = 9;
		//	*(h_c + 0) = 0;
		//	*(h_c + 1) = 0;
		//	*(h_c + 2) = 0;
		//	*(h_c + 3) = 0;
		//	*(h_c + 4) = 0;
		//	*(h_c + 5) = 0;
		//	*(h_c + 6) = 0;
		//	*(h_c + 7) = 0;
		//	*(h_c + 8) = 0;
		//	*(h_c + 9) = 0;
		//	//for (int row = 0; row < numRow; row++)
		//	//{
		//	//	for (int col = 0; col < numCol; col++)
		//	//	{
		//	//		//*(h_a + row * numCol + col) = 5;
		//	//		*(h_b + row * numCol + col) = 2;
		//	//		*(h_c + row * numCol + col) = 0;

		//	//	}
		//	//}

		//}



		//float data type
		void printMat(float* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%2.2f \t", *(h_a + row * numCol + col));
				}
				printf("\n");
			}
		}


		void linToMat(float* h_c, float h_clin[], int vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					*(h_c + row * numCol + col) = h_clin[vecIndex];
					vecIndex++;
				}
			}
		}


		void MatToLin(float* h_a, float* h_b, float* h_c, float h_alin[], float h_blin[], float h_clin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount] = *(h_a + row * numCol + col);
					h_blin[linCount] = *(h_b + row * numCol + col);
					h_clin[linCount] = *(h_c + row * numCol + col);
					linCount++;
				}
			}
		}



		void initMat(float* h_a, float* h_b, float* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					*(h_a + row * numCol + col) = 5;
					*(h_b + row * numCol + col) = 2;
					*(h_c + row * numCol + col) = 0;

				}
			}

		}

		//float2 data type
		void printMat(float2* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%2.2f \t", (*(h_a + row * numCol + col)).x);
				}
				printf("\n");
			}
		}


		void linToMat(float2* h_c, float2 h_clin[], int vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					(*(h_c + row * numCol + col)).x = h_clin[vecIndex].x;
					vecIndex++;
				}
			}
		}


		void MatToLin(float2* h_a, float2* h_b, float2* h_c, float2 h_alin[], float2 h_blin[], float2 h_clin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount].x = (*(h_a + row * numCol + col)).x;
					h_blin[linCount].x = (*(h_b + row * numCol + col)).x;
					h_clin[linCount].x = (*(h_c + row * numCol + col)).x;
					linCount++;
				}
			}
		}



		void initMat(float2* h_a, float2* h_b, float2* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					(*(h_a + row * numCol + col)).x = 5;
					(*(h_b + row * numCol + col)).x = 2;
					(*(h_c + row * numCol + col)).x = 0;

				}
			}

		}

		//float4
		void printMat(float4* h_a)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					printf("%2.2f \t", (*(h_a + row * numCol + col)).x);
				}
				printf("\n");
			}
		}


		void linToMat(float4* h_c, float4 h_clin[], int vecSize)
		{
			int vecIndex = 0;
			for (int col = 0; col < numCol; col++)
			{
				for (int row = 0; row < numRow; row++)
				{
					(*(h_c + row * numCol + col)).x = h_clin[vecIndex].x;
					vecIndex++;
				}
			}
		}


		void MatToLin(float4* h_a, float4* h_b, float4* h_c, float4 h_alin[], float4 h_blin[], float4 h_clin[], int vecSize)
		{
			int linCount = 0;
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					h_alin[linCount].x = (*(h_a + row * numCol + col)).x;
					h_blin[linCount].x = (*(h_b + row * numCol + col)).x;
					h_clin[linCount].x = (*(h_c + row * numCol + col)).x;
					linCount++;
				}
			}
		}



		void initMat(float4* h_a, float4* h_b, float4* h_c)
		{
			for (int row = 0; row < numRow; row++)
			{
				for (int col = 0; col < numCol; col++)
				{
					(*(h_a + row * numCol + col)).x = 5;
					(*(h_b + row * numCol + col)).x = 2;
					(*(h_c + row * numCol + col)).x = 0;

				}
			}

		}


		void getDeviceInfo()
		{
			cuInit(0);
			checkCuda(cuDeviceGetCount(&deviceCount));
			printf("Number of available CUDA devices: %d\n", deviceCount);

			for (int counter = 0; counter < deviceCount; counter++)
			{
				printf("Device #%d\n", counter);
				checkCuda(cuDeviceGetName(deviceName, 500, device));
				printf("device name: %s\n", deviceName);
				checkCuda(cuDeviceGetProperties(&deviceProp, counter));
			}
		}

		void printDeviceInfo()
		{
			getDeviceInfo();

			checkCuda(cuDeviceComputeCapability(&major, &minor, device));
			checkCuda(cuDriverGetVersion(&driver_version));
			checkCuda(cuDeviceTotalMem(&totalMemory, device));
			checkCuda(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
			checkCuda(cuDeviceGetAttribute(&memory_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
			checkCuda(cuDeviceGetAttribute(&global_memory_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));
			checkCuda(cuDeviceGetAttribute(&L2Cache, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
			checkCuda(cuDeviceGetAttribute(&max_shared_mem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
			checkCuda(cuDeviceGetAttribute(&max_reg_per_block, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
			checkCuda(cuDeviceGetAttribute(&warpsize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
			checkCuda(cuDeviceGetAttribute(&max_threads_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
			checkCuda(cuDeviceGetAttribute(&max_block_dimx, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
			checkCuda(cuDeviceGetAttribute(&max_block_dimy, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
			checkCuda(cuDeviceGetAttribute(&max_block_dimz, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
			checkCuda(cuDeviceGetAttribute(&max_grid_dimx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
			checkCuda(cuDeviceGetAttribute(&max_grid_dimy, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
			checkCuda(cuDeviceGetAttribute(&max_grid_dimz, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));

			printf("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: %d\n", major);
			printf("CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: %d\n", minor);
			printf("Cuda driver version: %d\n", driver_version);
			printf("CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY: %u\n", totalMemory);
			printf("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: %d\n", multiProcessorCount);
			printf("CU_DEVICE_ATTRIBUTE_CLOCK_RATE: %d\n", deviceProp.clockRate);
			printf("CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: %d\n", memory_clock_rate);
			printf("CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: %d\n", global_memory_bus_width);
			printf("CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: %d\n", L2Cache);//
			printf("CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: %d\n", deviceProp.totalConstantMemory);
			printf("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: %d\n", max_shared_mem_per_block);
			printf("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: %d\n", max_reg_per_block);
			printf("CU_DEVICE_ATTRIBUTE_WARP_SIZE: %d\n", warpsize);
			printf("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: %d\n", max_threads_per_multiprocessor);
			printf("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: %d\n", deviceProp.maxThreadsPerBlock);
			printf("CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: %d\n", max_block_dimx);
			printf("CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: %d\n", max_block_dimy);
			printf("CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: %d\n", max_block_dimz);
			printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: %d\n", max_grid_dimx);
			printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: %d\n", max_grid_dimy);
			printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: %d\n", max_grid_dimz);
		}


		int deviceCount = 0;
		char deviceName[500];
		int major = 0;
		int minor = 0;
		int driver_version = 0;
		int multiProcessorCount = 0;
		size_t totalMemory;
		int memory_clock_rate = 0;
		int global_memory_bus_width = 0;
		int L2Cache;
		int max_shared_mem_per_block = 0;
		int max_reg_per_block = 0;
		int warpsize = 0;
		int max_threads_per_multiprocessor = 0;
		int max_block_dimx = 0;
		int max_block_dimy = 0;
		int max_block_dimz = 0;
		int max_grid_dimx = 0;
		int max_grid_dimy = 0;
		int max_grid_dimz = 0;

		int numRow;
		int numCol;
		int numChannel;

		CUdevprop deviceProp;
		CUdevice device = 0;
	};

}






#endif

