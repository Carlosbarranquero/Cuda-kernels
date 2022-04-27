#include <stdlib.h>
#include <iostream>
using namespace std;


void print_host(float* vector, const int& size, const int& rows)
{
    const int cols = size/rows; // col_major

    for (int i=0; i< cols; i++)
    {
        for (int j=0; j< rows; j++)
        {
            cout<<*(vector + i + j*rows) << " ";
        }

        cout<<" "<<endl;
    }
}

__global__
void convolve_kernel(float* input_dev,float* kernel_dev, float* result_dev,const int result_rows_number, const int result_cols_number, const int input_dim, const int kernel_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_id = row*result_rows_number + col; // col major

    int offset = (input_dim - result_rows_number) * (thread_id/result_rows_number);
    
    float tmp_result = 0.0;

    for(int i=0; i<kernel_dim; i++)
    {
        for(int j=0; j<kernel_dim; j++)
        {
            tmp_result += input_dev[thread_id + offset + i*input_dim + j] * kernel_dev[i*kernel_dim + j];
        }
    }
    result_dev[thread_id] = tmp_result;
}


int main()
{
    // convolution between inputs and kernels
    cout<<"Hello kernels"<<endl;

    const int input_rows_number = 5;
    const int input_cols_number = 5;

    const int kernel_rows_number = 3; 
    const int kernel_cols_number = 3;

    const int result_rows_number = (input_rows_number - kernel_rows_number + 1);
    const int result_cols_number = (input_cols_number - kernel_cols_number + 1);

    const size_t input_size = input_cols_number*input_rows_number;
    const size_t kernel_size = kernel_cols_number*kernel_rows_number;
    const size_t result_size = result_rows_number*result_cols_number;

    //host 

    float* input_host = nullptr;
    float* kernel_host = nullptr;
    float* result_host = nullptr;

    input_host = (float*) malloc(input_size * sizeof(float));
    kernel_host = (float*) malloc(kernel_size * sizeof(float));
    result_host = (float*) malloc(result_size * sizeof(float));

    for (int i=0; i< input_size; i++)
        *(input_host + i) = i;

    for (int i=0; i< kernel_size; i++)
        *(kernel_host + i) = 2.0;

    for (int i=0; i< result_size; i++)
        *(result_host + i) = 0.0;

    cout<<"input host"<<endl;
    print_host(input_host, input_size, input_rows_number);
    cout<<""<<endl;
    cout<<"kernel host "<<endl;
    print_host(kernel_host, kernel_size, kernel_rows_number);
    cout<<""<<endl;
    
    //dev

    float* input_dev = nullptr;
    float* kernel_dev = nullptr;
    float* result_dev = nullptr;

    if(cudaMalloc(&input_dev, input_size*sizeof(float)) != cudaSuccess) cout << "Cuda malloc error" << endl;
    if(cudaMalloc(&kernel_dev, kernel_size*sizeof(float)) != cudaSuccess) cout << "Cuda malloc error" << endl;
    if(cudaMalloc(&result_dev, result_size*sizeof(float)) != cudaSuccess) cout << "Cuda malloc error" << endl;

    if(cudaMemcpy(input_dev, input_host, input_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Inputs copy error" << endl;
    if(cudaMemcpy(kernel_dev, kernel_host, kernel_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Inputs copy error" << endl;
    if(cudaMemcpy(result_dev, result_host, result_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "Inputs copy error" << endl;

    int threads = 32; // 64
    int blocks = (result_size + threads - 1 ) / threads;

    cout<<"threadsPerBlock: "<<threads<<endl;
    cout<<"blocksPerGrid: "<<blocks<<endl;

    dim3 threadsPerBlock(threads, threads);
    dim3 blocksPerGrid(blocks, blocks);

    convolve_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_dev, kernel_dev, result_dev, result_rows_number, result_cols_number, input_rows_number, kernel_rows_number);
    
    if(cudaMemcpy(result_host, result_dev, result_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Cuda matrix memcpy error" << endl;

    
    cout<<""<<endl;
        
     print_host(result_host, result_size, result_rows_number);
}
