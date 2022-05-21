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

int main()
{

    string test;

    cout << "Select test:"<<endl;

    cout << "1. Convolve_2d"<<endl;
    cout << "2. Convolve_3d"<<endl;
    cout << "3. Dot"<<endl;
    cout << "4. All"<<endl;
    cin >> test;


    if(test == "1" || test == "4")
    {    
    //  Test convolve 
        cout<<"<-----------------Test convolve----------------->"<<endl;

        const int input_rows_number = 4;
        const int input_cols_number = 4;

        const int kernel_rows_number = 2; 
        const int kernel_cols_number = 2;

        const int result_rows_number = (input_rows_number - kernel_rows_number + 1);
        const int result_cols_number = (input_cols_number - kernel_cols_number + 1);

        const size_t input_size = input_cols_number*input_rows_number;
        const size_t kernel_size = kernel_cols_number*kernel_rows_number;
        const size_t result_size = result_rows_number*result_cols_number;

        //host 

        float* input_host = nullptr;
        float* kernel_host = nullptr;
        float* result_host = nullptr;

        //Malloc

        if(cudaMallocHost(&input_host, input_size*sizeof(float)) != cudaSuccess)
            cout << "input_host allocation error" << endl;
        if(cudaMallocHost(&kernel_host, kernel_size*sizeof(float)) != cudaSuccess)
            cout << "kernel_host allocation error" << endl;
        if(cudaMallocHost(&result_host, result_size*sizeof(float)) != cudaSuccess)
            cout << "result_host allocation error" << endl;

        //Set values

        for (int i=0; i< input_size; i++)
            *(input_host + i) = i;

        for (int i=0; i< kernel_size; i++)
            *(kernel_host + i) = 2.0;

        for (int i=0; i< result_size; i++)
            *(result_host + i) = 0.0;

        //Print

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

        //Malloc

        if(cudaMalloc(&input_dev, input_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&kernel_dev, kernel_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&result_dev, result_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;

        //Cpy values H -> D

        if(cudaMemcpy(input_dev, input_host, input_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "input_dev error" << endl;
        if(cudaMemcpy(kernel_dev, kernel_host, kernel_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "kernel_dev error" << endl;
        if(cudaMemcpy(result_dev, result_host, result_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "result_dev error" << endl;

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
    else if(test == "2" || test == "4")
    {
        //  Test convolve 3d

        cout<<"<-----------------Test convolve 3d----------------->"<<endl;

        const int input_rows_number = 4;
        const int input_cols_number = 4;

        const int channels = 3;

        const int kernel_rows_number = 2; 
        const int kernel_cols_number = 2;

        const int result_rows_number = (input_rows_number - kernel_rows_number + 1);
        const int result_cols_number = (input_cols_number - kernel_cols_number + 1);

        const size_t input_size = input_cols_number*input_rows_number*channels;
        const size_t kernel_size = kernel_cols_number*kernel_rows_number*channels;
        const size_t result_size = result_rows_number*result_cols_number;

        //host 

        float* input_host = nullptr;
        float* kernel_host = nullptr;
        float* result_host = nullptr;

        //Malloc

        if(cudaMallocHost(&input_host, input_size*sizeof(float)) != cudaSuccess)
            cout << "input_host allocation error" << endl;
        if(cudaMallocHost(&kernel_host, kernel_size*sizeof(float)) != cudaSuccess)
            cout << "kernel_host allocation error" << endl;
        if(cudaMallocHost(&result_host, result_size*sizeof(float)) != cudaSuccess)
            cout << "result_host allocation error" << endl;

        //Set values

        for (int i=0; i< input_size; i++)
            *(input_host + i) = 1.;

        for (int i=0; i< kernel_size; i++)
            *(kernel_host + i) = 2.;

        for (int i=0; i< result_size; i++)
            *(result_host + i) = 0.0;

        //Print

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

        //Malloc

        if(cudaMalloc(&input_dev, input_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&kernel_dev, kernel_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&result_dev, result_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;

        //Cpy values H -> D

        if(cudaMemcpy(input_dev, input_host, input_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "input_dev error" << endl;
        if(cudaMemcpy(kernel_dev, kernel_host, kernel_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "kernel_dev error" << endl;
        if(cudaMemcpy(result_dev, result_host, result_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "result_dev error" << endl;

        int threads = 32; // 64
        int blocks = (result_size + threads - 1 ) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock(threads, threads);
        dim3 blocksPerGrid(blocks, blocks);

        time_t beginning_time, current_time;
        time(&beginning_time);
        float elapsed_time = 0;

        
        convolve_kernel_3d<<<blocksPerGrid, threadsPerBlock>>>(input_dev, kernel_dev, result_dev, result_rows_number, result_cols_number, input_rows_number, kernel_rows_number, channels);
        
        
        if(cudaMemcpy(result_host, result_dev, result_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "Cuda matrix memcpy error" << endl;

        cout<<"end"<<endl;

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        cout<<"elapsed_time:"<< elapsed_time<<endl;
            
        print_host(result_host, result_size, result_rows_number);
        
    }
    else if(test == "3" || test == "4")
    {
        //  Test dot A*B  = C

        cout<<""<<endl;
        cout<<"<-----------------Test dot----------------->"<<endl;

        const int A_rows_number = 30;
        const int A_cols_number = 30;
        const int A_size = A_rows_number*A_cols_number;

        const int B_rows_number = 20;
        const int B_cols_number = 20;
        const int B_size = B_rows_number*B_cols_number;

        const int C_rows_number = A_rows_number;
        const int C_cols_number = B_cols_number;
        const int C_size = C_rows_number*C_cols_number;

        //host 

        float* A_host = nullptr;
        float* B_host = nullptr;
        float* C_host = nullptr;

        //Malloc

        if(cudaMallocHost(&A_host, A_size*sizeof(float)) != cudaSuccess)
            cout << "A_host allocation error" << endl;
        if(cudaMallocHost(&B_host, B_size*sizeof(float)) != cudaSuccess)
            cout << "B_host allocation error" << endl;
        if(cudaMallocHost(&C_host, C_size*sizeof(float)) != cudaSuccess)
            cout << "C_host allocation error" << endl;

        //Set values

        for (int i=0; i< A_size; i++)
            *(A_host + i) = rand() % 10;

        for (int i=0; i< B_size; i++)
            *(B_host + i) = rand() % 10;

        for (int i=0; i< C_size; i++)
            *(C_host + i) = 0.0;

        //Print

        cout<<"A host"<<endl;
        print_host(A_host, A_size, A_rows_number);
        cout<<""<<endl;
        cout<<"B host "<<endl;
        print_host(B_host, B_size, B_rows_number);
        cout<<""<<endl;

        //dev

        float* A_dev = nullptr;
        float* B_dev = nullptr;
        float* C_dev = nullptr;

        //Malloc

        if(cudaMalloc(&A_dev, A_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&B_dev, B_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;
        if(cudaMalloc(&C_dev, C_size*sizeof(float)) != cudaSuccess)
            cout << "Cuda malloc error" << endl;

        //Cpy values H -> D

        if(cudaMemcpy(A_dev, A_host, A_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "A_dev copy error" << endl;
        if(cudaMemcpy(B_dev, B_host, B_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "B_dev copy error" << endl;
        if(cudaMemcpy(C_dev, C_host, C_size*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "C_dev copy error" << endl;

        int threads = 32; // 64
        int blocks = (C_size + threads - 1 ) / threads;

        cout<<"threadsPerBlock: "<<threads<<endl;
        cout<<"blocksPerGrid: "<<blocks<<endl;

        dim3 threadsPerBlock_dot(threads, threads);
        dim3 blocksPerGrid_dot(blocks, blocks);

        dot_kernel<<<blocksPerGrid_dot, threadsPerBlock_dot>>>(A_dev, B_dev, C_dev, B_rows_number, C_rows_number);

        if(cudaMemcpy(C_host, C_dev, C_size*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "C memcpy error" << endl;

        cout<<""<<endl;
            
        print_host(C_host, C_size, C_rows_number);
    }
}
