#include <stdlib.h>
#include <iostream>
#include "cuda_operations.cu"
using namespace std;

void print_host(float* vector, const int& size, const int& rows)
{
    const int cols = size/rows; // col_major

    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            cout<<*(vector + i + rows*j )<< " ";
        }

        cout<<" "<<endl;
    }
}

float* build_host_tensor(const int dims[], const float& values, const bool& randomize)
{
    const int rows_number = dims[0];
    const int cols_number = dims[1];
    const int channels = dims[2];

    const int size = rows_number*cols_number*channels;
    
    // Initialize
    float* A_host = nullptr;
    
    // Malloc
    if(cudaMallocHost(&A_host, size*sizeof(float)) != cudaSuccess)
        cout << "A_host allocation error" << endl;
    
    // Set Values    
    for (int i=0; i< size; i++) 
        *(A_host + i) = randomize ? rand() % 10: values;
    
    // Print 
    bool is_printed = randomize ? true: false;

    if(is_printed)
    {
        cout<<"Tensor info"<< endl;
        cout<<"Rows_number: "<< rows_number<<endl;
        cout<<"Cols_number: "<< cols_number<<endl;
        cout<<"Channels_number: "<< channels<<endl;
        cout<<"---"<<endl;
        print_host(A_host, size, rows_number);
    }

    return A_host;
}


float* copy_host_tensor_to_dev(const int dims[], float* A_host)
{
    const int rows_number = dims[0];
    const int cols_number = dims[1];
    const int channels = dims[2];

    const int size = rows_number*cols_number*channels;

    // Initialize
    float* A_dev = nullptr;

    // Malloc
    if(cudaMalloc(&A_dev, size*sizeof(float)) != cudaSuccess)
        cout << "Cuda malloc error" << endl;

    //Cpy values H -> D
    if(cudaMemcpy(A_dev, A_host, size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "A_dev copy error" << endl;

    return A_dev;
}


int main()
{

    string test;

    cout << "Select test:"<<endl;

    cout << "0. All"<<endl;
    cout << "1. Convolve_2d"<<endl;
    cout << "2. Convolve_3d"<<endl;
    cout << "3. Dot"<<endl;
    cout << "4. Softmax"<<endl;
    
    cin >> test;

    if(test == "1" || test == "0")
    {    
        cout<<"<-----------------Test convolve 2d----------------->"<<endl;

        const int input_dims[] = {4, 4, 1}; // rows, cols, channels
        const int kernel_dims[] = {2, 2, 1}; // rows, cols, channels
        const int results_dims[] = {input_dims[0] - kernel_dims[0] + 1,
                                    input_dims[1] - kernel_dims[1] + 1,
                                    input_dims[2]};

        const float initial_value = 0.0;

        // Host
        float* input_host = build_host_tensor(input_dims, initial_value, true);
        float* kernel_host = build_host_tensor(kernel_dims, initial_value, true);
        float* result_host = build_host_tensor(results_dims, initial_value, false);
        
        // Dev
        float* input_dev = copy_host_tensor_to_dev(input_dims, input_host);
        float* kernel_dev = copy_host_tensor_to_dev(kernel_dims, kernel_host);
        float* result_dev = copy_host_tensor_to_dev(results_dims, result_host);

        // Grid 
        const int result_size = results_dims[0]*results_dims[1]*results_dims[2];
        const int threads = 32; // 64
        const int blocks = (result_size + threads - 1) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock(threads, threads);
        dim3 blocksPerGrid(blocks, blocks);

        convolve_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_dev, kernel_dev, result_dev, results_dims[0], results_dims[1], input_dims[0], kernel_dims[0]);
        
        if(cudaMemcpy(result_host, result_dev, result_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "Cuda matrix memcpy error" << endl;

        cout<<""<<endl;

        cout<<"Results"<<endl;    
        print_host(result_host, result_size, results_dims[0]);

    }
    else if(test == "2" || test == "0")
    {
        cout<<"<-----------------Test convolve 3d----------------->"<<endl;

        const int input_dims[] = {4, 4, 3}; // rows, cols, channels
        const int kernel_dims[] = {2, 2, 3}; // rows, cols, channels
        const int results_dims[] = {input_dims[0] - kernel_dims[0] + 1,
                                    input_dims[1] - kernel_dims[1] + 1,
                                    1};

        const float initial_value = 0.0;

        // Host
        float* input_host = build_host_tensor(input_dims, initial_value, true);
        float* kernel_host = build_host_tensor(kernel_dims, initial_value, true);
        float* result_host = build_host_tensor(results_dims, initial_value, false);
        
        // Dev
        float* input_dev = copy_host_tensor_to_dev(input_dims, input_host);
        float* kernel_dev = copy_host_tensor_to_dev(kernel_dims, kernel_host);
        float* result_dev = copy_host_tensor_to_dev(results_dims, result_host);

        // Grid 
        const int result_size = results_dims[0]*results_dims[1]*results_dims[2];
        const int threads = 32; // 64
        const int blocks = (result_size + threads - 1 ) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock(threads, threads);
        dim3 blocksPerGrid(blocks, blocks);

        time_t beginning_time, current_time;
        time(&beginning_time);
        float elapsed_time = 0;

        convolve_kernel_3d<<<blocksPerGrid, threadsPerBlock>>>(input_dev, kernel_dev, result_dev, results_dims[0], results_dims[1], input_dims[0], kernel_dims[0], input_dims[2]);
        
        if(cudaMemcpy(result_host, result_dev, result_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "Cuda matrix memcpy error" << endl;

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        cout<<"elapsed_time:"<< elapsed_time<<endl;

        cout<<""<<endl;    
        print_host(result_host, result_size, results_dims[0]);

    }
    else if(test == "3" || test == "0")
    {
        
        cout<<"<-----------------Test dot A*B = C----------------->"<<endl;

        // Host
        const float initial_value_A = 1.0;
        const int A_dims[] = {2, 2, 1}; // rows, cols, channels
        float* A_host = build_host_tensor(A_dims, initial_value_A, true);

        const float initial_value_B = 2.0;
        const int B_dims[] = {2, 2, 1}; // rows, cols, channels
        float* B_host = build_host_tensor(B_dims, initial_value_B, true);

        const float initial_value_C = 0.0;
        const int C_dims[] = {A_dims[0], B_dims[0], 1}; // rows, cols, channels
        float* C_host = build_host_tensor(C_dims, initial_value_C, false);

        // Dev
        float* A_dev = copy_host_tensor_to_dev(A_dims, A_host);
        float* B_dev = copy_host_tensor_to_dev(B_dims, B_host);
        float* C_dev = copy_host_tensor_to_dev(C_dims, C_host);

        // Grid 
        const int result_size = C_dims[0]*C_dims[1]*C_dims[2];
        const int threads = 32; // 64
        const int blocks = (result_size + threads - 1 ) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock(threads, threads);
        dim3 blocksPerGrid(blocks, blocks);

        time_t beginning_time, current_time;
        time(&beginning_time);
        float elapsed_time = 0;

        dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev, B_dims[0], A_dims[0]);
        
        if(cudaMemcpy(C_host, C_dev, result_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "Cuda matrix memcpy error" << endl;

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        cout<<"elapsed_time:"<< elapsed_time<<endl;

        cout<<""<<endl; 
        print_host(C_host, result_size, C_dims[0]);   

    }
    else if(test == "4" || test == "0")
    {
        cout<<"<-----------------Test Softmax----------------->"<<endl;

        const int dims[] = {2, 2, 1}; // rows, cols, channels
        const float initial_value = 0.0;

        // Host
        float* A_host = build_host_tensor(dims, initial_value, true);
        float* B_host = build_host_tensor(dims, initial_value, false);
        
        // Dev
        float* A_dev = copy_host_tensor_to_dev(dims, A_host);
        float* B_dev = copy_host_tensor_to_dev(dims, A_host);

        // Grid
        const int size = dims[0]*dims[1]*dims[2];
        int threads = 32; // 64
        int blocks = (size + threads - 1 ) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock(threads, threads);
        dim3 blocksPerGrid(blocks, blocks);

        softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, dims[0], A_dev, B_dev);

        if(cudaMemcpy(B_host, B_dev, size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "C memcpy error" << endl;

        print_host(B_host, size, dims[0]);
    }

}
