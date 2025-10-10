/**
 * Vector Addition Kernel
 *
 * Adds two vectors: C = A + B
 */

__global__ void vecadd_kernel(const float* A, const float* B, float* C, int N) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
