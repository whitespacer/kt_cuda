#include "gpu_timer.h"

#include <vector>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define FILTER_WIDTH 11
#define FILTER_RADIUS FILTER_WIDTH / 2
#define BLOCK_WIDTH 16

#define M_PI_F 3.14159265358979323846f

namespace {
    __constant__ float g_const_filter[FILTER_WIDTH*FILTER_WIDTH];
    enum class mode_t
    {
        simple,
        simple_with_const_filter,
        shared
    };
}


void convolution_cpu(std::vector<unsigned char> & out_img,
    unsigned char const* in_img, std::vector<float> const& filter,
    int num_cols, int num_rows, int num_channels)
{
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            for (int channel = 0; channel < num_channels; channel++) {
                float sum = 0;

                // accumulate values inside filter
                for (int i = 0; i < FILTER_WIDTH; i++) {
                    for (int j = 0; j < FILTER_WIDTH; j++) {
                        int img_row = row - FILTER_RADIUS + i;
                        int img_col = col - FILTER_RADIUS + j;

                        if ((img_row >= 0) && (img_row < num_rows) && (img_col >= 0) && (img_col < num_cols)) {
                            sum += in_img[(img_row*num_cols + img_col)*num_channels + channel] * filter[i*FILTER_WIDTH + j];
                        }
                    }
                }
                out_img[(row*num_cols + col)*num_channels + channel] = (unsigned char)sum;
            }
        }
    }
}

__global__ void convolution_gpu_simple(unsigned char* out_img, unsigned char* in_img, const float* filter,
    int num_cols, int num_rows, int num_channels)
{
    // one thread computes convolution for one pixel

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_rows || col >= num_cols)
        return;

    // loop through the channels
    for (int channel = 0; channel < num_channels; channel++) {
        float sum = 0;

        for (int i = 0; i < FILTER_WIDTH; i++) {
            for (int j = 0; j < FILTER_WIDTH; j++) {
                int img_row = row - FILTER_RADIUS + i;
                int img_col = col - FILTER_RADIUS + j;

                if ((img_row >= 0) && (img_row < num_rows) && (img_col >= 0) && (img_col < num_cols)) {
                    sum += in_img[(img_row*num_cols + img_col)*num_channels + channel] * filter[i*FILTER_WIDTH + j];
                }
            }
        }
        out_img[(row*num_cols + col)*num_channels + channel] = sum;
    }
}

__global__ void convolution_gpu_simple_const(unsigned char* out_img, unsigned char* in_img,
    int num_cols, int num_rows, int num_channels)
{
    // one thread computes convolution for one pixel

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= num_rows || col >= num_cols)
        return;

    // loop through the channels
    for (int channel = 0; channel < num_channels; channel++) {
        float sum = 0;

        for (int i = 0; i < FILTER_WIDTH; i++) {
            for (int j = 0; j < FILTER_WIDTH; j++) {
                int img_row = row - FILTER_RADIUS + i;
                int img_col = col - FILTER_RADIUS + j;

                if ((img_row >= 0) && (img_row < num_rows) && (img_col >= 0) && (img_col < num_cols)) {
                    sum += in_img[(img_row*num_cols + img_col)*num_channels + channel] * g_const_filter[i*FILTER_WIDTH + j];
                }
            }
        }
        out_img[(row*num_cols + col)*num_channels + channel] = sum;
    }
}

__global__ void convolution_gpu_shared(unsigned char* out_img, unsigned char const* in_img,
                                       int num_cols, int num_rows, int num_channels)
{
    // one thread computes convolution for one pixel

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // loop through the channels
    for (int channel = 0; channel < num_channels; channel++) {
        __shared__ unsigned char data[BLOCK_WIDTH + 2 * FILTER_RADIUS][BLOCK_WIDTH + 2 * FILTER_RADIUS];

        int row2 = row - FILTER_RADIUS;
        int col2 = col - FILTER_RADIUS;
        if (row2 < num_rows && col2 < num_cols && row2 > 0 && col2 > 0)
            data[threadIdx.y][threadIdx.x] = in_img[(row2 * num_cols + col2)*num_channels + channel];

        int row3 = row - FILTER_RADIUS + BLOCK_WIDTH;
        int col3 = col - FILTER_RADIUS + BLOCK_WIDTH;
        if (row3 < num_rows && col3 < num_cols && threadIdx.y < 2 * FILTER_RADIUS && threadIdx.x < 2 * FILTER_RADIUS)
            data[threadIdx.y + BLOCK_WIDTH][threadIdx.x + BLOCK_WIDTH] = in_img[(row3 * num_cols + col3)*num_channels + channel];

        int row4 = row - FILTER_RADIUS;
        int col4 = col - FILTER_RADIUS + BLOCK_WIDTH;
        if (row4 < num_rows && col4 < num_cols && row4 > 0 && threadIdx.x < 2 * FILTER_RADIUS)
            data[threadIdx.y][threadIdx.x + BLOCK_WIDTH] = in_img[(row4 * num_cols + col4)*num_channels + channel];

        int row5 = row - FILTER_RADIUS + BLOCK_WIDTH;
        int col5 = col - FILTER_RADIUS;
        if (row5 < num_rows && col5 < num_cols && col5 > 0 && threadIdx.y < 2 * FILTER_RADIUS)
            data[threadIdx.y + BLOCK_WIDTH][threadIdx.x] = in_img[(row5 * num_cols + col5)*num_channels + channel];

        __syncthreads();

        float sum = 0;

        for (int i = 0; i < FILTER_WIDTH; i++) {
            for (int j = 0; j < FILTER_WIDTH; j++) {
                sum += data[threadIdx.y + i][threadIdx.x + j] * g_const_filter[i*FILTER_WIDTH + j];
            }
        }

        if (row < num_rows && col < num_cols)
            out_img[(row*num_cols + col)*num_channels + channel] = sum;

        __syncthreads();
    }
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        std::cerr << "Provide image filename as first argument" << std::endl;
        return 1;
    }

    // read image
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image file " << argv[1] << std::endl;
        exit(1);
    }

    mode_t mode = mode_t::shared;
    if (argc == 3) {
        std::string requested_mode(argv[2]);
        if (requested_mode == "simple") {
            mode = mode_t::simple;
        }
        else if (requested_mode == "simple_with_const_filter") {
            mode = mode_t::simple_with_const_filter;
        }
        else if (requested_mode == "shared") {
            mode = mode_t::shared;
        }
    }

    // grab image dimensions
    int img_channels = img.channels();
    int img_width = img.cols;
    int img_height = img.rows;

    // useful params
    gpu_timer_t timer0, timer1, timer2, timer3;

    // allocate host memory
    std::vector<float> host_filter(FILTER_WIDTH*FILTER_WIDTH);
    std::vector<unsigned char> host_out_img(img_width*img_height*img_channels);

    // hardcoded filter values
    /*
    float filter[FILTER_WIDTH*FILTER_WIDTH] = {
    1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0,
    4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
    7 / 273.0, 26 / 273.0, 41 / 273.0, 26 / 273.0, 7 / 273.0,
    4 / 273.0, 16 / 273.0, 26 / 273.0, 16 / 273.0, 4 / 273.0,
    1 / 273.0, 4 / 273.0, 7 / 273.0, 4 / 273.0, 1 / 273.0
    };
    std::copy(filter, filter + FILTER_WIDTH*FILTER_WIDTH, &host_filter[0]);
    */

    float sigma = 10;
    float sum = 0;
    for (int i = 0; i < FILTER_WIDTH; i++) {
        for (int j = 0; j < FILTER_WIDTH; j++) {
            float val = exp(-(i*i + j*j) / (2 * sigma*sigma)) / (2 * M_PI_F*sigma*sigma);
            host_filter[i*FILTER_WIDTH + j] = val;
            sum += val;
        }
    }
    for (int i = 0; i < FILTER_WIDTH*FILTER_WIDTH; i++)
        host_filter[i] /= sum;

    // allocate device memory
    float* dev_filter = nullptr;
    unsigned char* dev_in_img = nullptr;
    unsigned char* dev_out_img = nullptr;
    timer0.start();
    size_t img_size_in_bytes = sizeof(unsigned char)*img_width*img_height*img_channels;
    size_t filter_size_in_bytes = sizeof(float)*FILTER_WIDTH*FILTER_WIDTH;
    cudaMalloc((void**)&dev_filter, filter_size_in_bytes);
    cudaMalloc((void**)&dev_in_img, img_size_in_bytes);
    cudaMalloc((void**)&dev_out_img, img_size_in_bytes);
    timer0.stop();
    float t0 = timer0.elapsed();
    std::cout << "Time to allocate memory on divice in msecs: " << t0 << std::endl;

    // host2device transfer
    timer1.start();
    cudaMemcpy(dev_filter, &host_filter[0], filter_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_in_img, img.data, img_size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_const_filter, &host_filter[0], filter_size_in_bytes);
    timer1.stop();
    float t1 = timer1.elapsed();
    std::cout << "Time for host to device transfer in msecs: " << t1 << std::endl;

    // kernel launch
    timer2.start();

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid((img_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (img_height + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
    switch (mode) {
    case mode_t::simple:
        convolution_gpu_simple << < dimGrid, dimBlock >> > (dev_out_img, dev_in_img, dev_filter, img_width, img_height, img_channels);
        break;
    case mode_t::simple_with_const_filter:
        convolution_gpu_simple_const << < dimGrid, dimBlock >> > (dev_out_img, dev_in_img, img_width, img_height, img_channels);
        break;
    case mode_t::shared:
        convolution_gpu_shared << < dimGrid, dimBlock >> > (dev_out_img, dev_in_img, img_width, img_height, img_channels);
    }

    timer2.stop();
    float t2 = timer2.elapsed();
    std::cout << "Time for kernel run in msecs: " << t2 << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to launch convolution kernel" << cudaGetErrorString(err);
        exit(1);
    }

    // device2host transfer
    timer3.start();
    cudaMemcpy(&host_out_img[0], dev_out_img, img_size_in_bytes, cudaMemcpyDeviceToHost);
    timer3.stop();
    float t3 = timer3.elapsed();
    std::cout << "Time for device to host transfer in msecs: " << t3 << std::endl;

    // do the processing on the CPU
    std::vector<unsigned char> host_out_img_cpu(img_size_in_bytes);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    convolution_cpu(host_out_img_cpu, img.data, host_filter, img_width, img_height, img_channels);
    auto stop_cpu = std::chrono::high_resolution_clock::now();

    // calculate total time for CPU and GPU
    auto time_spent_cpu = std::chrono::duration <double, std::milli>(stop_cpu - start_cpu).count();
    std::cout << "Total CPU time in msec: " << time_spent_cpu << std::endl;
    float time_spent_gpu = t0 + t1 + t2 + t3;
    std::cout << "Total GPU time in msec: " << time_spent_gpu << std::endl;
    float speedup = (float)time_spent_cpu / time_spent_gpu;
    std::cout << "GPU Speedup: " << speedup << std::endl;

    // display images
    cv::Mat img_gpu(img_height, img_width, CV_8UC3, &host_out_img[0]);
    cv::Mat img_cpu(img_height, img_width, CV_8UC3, &host_out_img_cpu[0]);
    cv::namedWindow("Before", cv::WINDOW_NORMAL);
    cv::resizeWindow("Before", img_width, img_height);
    cv::imshow("Before", img);
    cv::namedWindow("After (GPU)", cv::WINDOW_NORMAL);
    cv::resizeWindow("After (GPU)", img_width, img_height);
    cv::imshow("After (GPU)", img_gpu);
    cv::namedWindow("After (CPU)", cv::WINDOW_NORMAL);
    cv::resizeWindow("After (CPU)", img_width, img_height);
    cv::imshow("After (CPU)", img_cpu);
    cv::waitKey(0);

    // free host and device memory
    img.release();
    img_gpu.release();
    img_cpu.release();

    cudaFree(dev_out_img);
    cudaFree(dev_in_img);
    cudaFree(dev_filter);

    return 0;
}