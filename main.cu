#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <memory>
#include <functional>
#include <cstring>
#include <boost/program_options.hpp>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

__host__ __device__ inline int offset(int row, int col, int num_cols) {
    return row * num_cols + col;
}

namespace po = boost::program_options;

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

template <typename T>
T *cuda_new(std::size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}

template <typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr);
}

__global__ void grid_kernel(double* A, double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        Anew[offset(i, j, size)] = 0.25 * (A[offset(i, j + 1, size)] + A[offset(i, j - 1, size)] +
                                          A[offset(i + 1, j, size)] + A[offset(i - 1, j, size)]);
    }
}

__global__ void error_kernel(double* A, double* Anew, double* block_max_errors, int size) {
    double inner_error = 0.0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int idx = thread_id; idx < size * size; idx += total_threads) {
        int i = idx / size;
        int j = idx % size;
        if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
            inner_error = fmax(inner_error, fabs(Anew[offset(i, j, size)] - A[offset(i, j, size)]));
        }
    }

    double block_max = BlockReduce(temp_storage).Reduce(inner_error, cub::Max());
    if (threadIdx.x == 0)
        block_max_errors[blockIdx.x] = block_max;
}

void initialize(double* A, double* Anew, int rows, int cols) {
    memset(A, 0, sizeof(double) * rows * cols);
    memset(Anew, 0, sizeof(double) * rows * cols);

    double topLeft = 10.0;
    double topRight = 20.0;
    double bottomLeft = 20.0;
    double bottomRight = 30.0;

    for (int j = 0; j < cols; ++j) {
        double alpha = static_cast<double>(j) / (cols - 1);
        A[offset(0, j, cols)] = Anew[offset(0, j, cols)] = (1 - alpha) * topLeft + alpha * topRight;
    }

    for (int j = 0; j < cols; ++j) {
        double alpha = static_cast<double>(j) / (cols - 1);
        A[offset(rows - 1, j, cols)] = Anew[offset(rows - 1, j, cols)] = (1 - alpha) * bottomLeft + alpha * bottomRight;
    }

    for (int i = 0; i < rows; ++i) {
        double alpha = static_cast<double>(i) / (rows - 1);
        A[offset(i, 0, cols)] = Anew[offset(i, 0, cols)] = (1 - alpha) * topLeft + alpha * bottomLeft;
    }

    for (int i = 0; i < rows; ++i) {
        double alpha = static_cast<double>(i) / (rows - 1);
        A[offset(i, cols - 1, cols)] = Anew[offset(i, cols - 1, cols)] = (1 - alpha) * topRight + alpha * bottomRight;
    }
}

int main(int argc, char** argv) {
    int n, m, iter_max;
    double tol;

    po::options_description desc("Опции");
    desc.add_options()
        ("help", "показать справку")
        ("size", po::value<int>(&n)->default_value(512), "размер матрицы (n x n)")
        ("tol", po::value<double>(&tol)->default_value(1.0e-6), "точность")
        ("max_iter", po::value<int>(&iter_max)->default_value(1000000), "максимум итераций");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    m = n;
    double error = 1.0;
    int iteration = 0;

    std::unique_ptr<double[]> A = std::make_unique<double[]>(n * m);
    std::unique_ptr<double[]> Anew = std::make_unique<double[]>(n * m);

    initialize(A.get(), Anew.get(), m, n);

    double* d_A = cuda_new<double>(n * m);
    double* d_Anew = cuda_new<double>(n * m);
    cudaMemcpy(d_A, A.get(), sizeof(double) * n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew.get(), sizeof(double) * n * m, cudaMemcpyHostToDevice);

    cuda_unique_ptr<double> d_matA(d_A, cuda_delete<double>);
    cuda_unique_ptr<double> d_matB(d_Anew, cuda_delete<double>);

    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((n + threads - 1) / threads, (m + threads - 1) / threads);

    int num_blocks = 1024;
    double* d_block_max;
    cudaMalloc(&d_block_max, sizeof(double) * num_blocks);

    std::unique_ptr<double[]> h_block_max = std::make_unique<double[]>(num_blocks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < 1000; ++i) {
        grid_kernel<<<grid, block, 0, stream>>>(d_A, d_Anew, n);
        std::swap(d_A, d_Anew); 
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    std::cout << "Сетка: " << n << " x " << m << "\n";

    const auto start = std::chrono::steady_clock::now();

    while (error > tol && iteration < iter_max) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        error_kernel<<<num_blocks, 256, 0, stream>>>(d_A, d_Anew, d_block_max, n);
        cudaMemcpy(h_block_max.get(), d_block_max, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < num_blocks; ++i)
            error = std::max(error, h_block_max[i]);

        iteration += 1000;
        std::swap(d_A, d_Anew);
    }

    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Итераций: " << iteration << "\n";
    std::cout << "Ошибка: " << error << "\n";
    std::cout << "Время: " << elapsed.count() << " секунд\n";

    cudaFree(d_block_max);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);

    return 0;
}
