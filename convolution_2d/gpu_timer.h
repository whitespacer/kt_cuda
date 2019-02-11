#pragma once

#include <cuda_runtime.h>

struct gpu_timer_t
{
    gpu_timer_t()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~gpu_timer_t()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start()
    {
        cudaEventRecord(start_, 0);
    }

    void stop()
    {
        cudaEventRecord(stop_, 0);
    }

    float elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
