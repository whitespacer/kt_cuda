#include <cuda_runtime.h>

#include <vector>
#include <iostream>


__global__ void vector_add(const float *a, const float *b, float *c, int num_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    size_t const num_elements = 5000000;
    size_t const elements_size = num_elements * sizeof(float);

    // allocate cpu data
    std::vector<float> host_a(num_elements);
    std::vector<float> host_b(num_elements);
    std::vector<float> host_c(num_elements);

    for (size_t i = 0; i < num_elements; ++i)
    {
        host_a[i] = rand() / (float)RAND_MAX;
        host_b[i] = rand() / (float)RAND_MAX;
    }

    // allocate device data
    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;

    cudaMalloc((void **)&dev_a, elements_size);
    cudaMalloc((void **)&dev_b, elements_size);
    cudaMalloc((void **)&dev_c, elements_size);

    // Copy "A" and "B" from host to device
    cudaMemcpy(dev_a, &host_a[0], elements_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &host_b[0], elements_size, cudaMemcpyHostToDevice);

    // launch_kernel
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    vector_add <<< num_blocks, block_size >>>(dev_a, dev_b, dev_c, num_elements);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch vectorAdd kernel" << cudaGetErrorString(err);
        exit(1);
    }

    // Copy "C" from device to host
    cudaMemcpy(&host_c[0], dev_c, elements_size, cudaMemcpyDeviceToHost);

    // verify result
    for (size_t i = 0; i < num_elements; ++i)
    {
        if (fabs(host_a[i] + host_b[i] - host_c[i]) > 1e-5)
        {
            std::cerr << "Failure at " << i << std::endl;
            exit(1);
        }
    }

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::cout << "Done" << std::endl;
    return 0;
}

