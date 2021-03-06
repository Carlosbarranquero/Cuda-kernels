__global__
void dot_kernel(float* A, float* B, float* C, const int B_rows_number, const int C_rows_number)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_id = row*C_rows_number + col; // col major

    float tmpSum = 0;

    if(thread_id < B_rows_number*C_rows_number)
    {
        for(int i=0; i<B_rows_number; i++)
        {
            tmpSum += A[thread_id%C_rows_number + i*C_rows_number] * B[thread_id/C_rows_number*B_rows_number + i];
        }

        C[thread_id] = tmpSum;    
    }
}


__global__
void convolve_kernel(float* input_dev,float* kernel_dev, float* result_dev,const int result_rows_number, const int result_cols_number, const int input_dim, const int kernel_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_id = row*result_rows_number + col; // col major

    if(thread_id < result_cols_number*result_rows_number)
    {
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

}


__global__
void convolve_kernel_3d(const float* input_dev, const float* kernel_dev, float* result_dev, const int result_rows_number, const int result_cols_number, const int input_dim, const int kernel_dim, const int channels)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_id = row*result_rows_number + col; // col major

    int input_size = (result_rows_number +  kernel_dim - 1)*(result_rows_number +  kernel_dim - 1);

    if(thread_id < result_cols_number*result_rows_number)
    {
        int offset = (input_dim - result_rows_number) * (thread_id/result_rows_number);

        float tmp_result = 0.0;

        for(int k=0; k < channels; k++) // this can be optimazed with share memory 
        {
            for(int i=0; i<kernel_dim; i++)
            {
                for(int j=0; j<kernel_dim; j++)
                {
                    tmp_result += input_dev[thread_id + offset + i*input_dim + j + input_size*k] * kernel_dev[i*kernel_dim + j + kernel_dim*kernel_dim*k];
                }
            }
        }
        result_dev[thread_id] = tmp_result;
    }

}

__global__
void softmax_kernel(int n, int rows_number, float* inputs, float* outputs)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_id = row*rows_number + col; // col major

    const int columns_number = n/rows_number;

    if(thread_id < rows_number*columns_number) 
    {
        float sum = 0;

        for(int j = 0; j < columns_number; j++)
        {
            sum += inputs[thread_id%rows_number + rows_number*j];
        }

        outputs[thread_id] = inputs[thread_id]/sum;
    }

}
