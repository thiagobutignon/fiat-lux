/**
 * Matrix Multiplication Kernel
 *
 * Computes C = A * B
 * where A is MxK, B is KxN, C is MxN
 */

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (row < M && col < N) {
        float sum = 0.0f;

        // Dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Write result to C
        C[row * N + col] = sum;
    }
}
