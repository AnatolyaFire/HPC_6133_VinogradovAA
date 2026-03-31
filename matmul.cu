// Умножение матриц: CPU vs GPU (CUDA)

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

const int TILE_SIZE = 16;

// CUDA ядро для умножения матриц с использованием разделяемой памяти
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    // Разделяемая память для подматриц
    __shared__ double sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ double sharedB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Цикл по всем подматрицам
    for (int tile = 0; tile < numTiles; ++tile) {
        // Загрузка данных в разделяемую память
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            sharedA[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0;
        }
        
        if (col < N && (tile * TILE_SIZE + ty) < N) {
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Умножение элементов подматрицы
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }
    
    // Запись результата
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Функция умножения матриц на CPU
void multiplyMatricesCPU(const vector<double>& A, 
                         const vector<double>& B, 
                         vector<double>& C, 
                         int N) {
    fill(C.begin(), C.end(), 0.0);
    
    // Оптимизированный порядок циклов i-k-j для улучшения кэш-локальности
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

// Функция умножения матриц на GPU
void multiplyMatricesGPU(const vector<double>& A, 
                         const vector<double>& B, 
                         vector<double>& C, 
                         int N) {
    double *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(double);
    
    // Выделение памяти на GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Копирование данных на GPU
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);
    
    // Настройка сетки блоков и потоков
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Запуск CUDA ядра
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Копирование результата обратно на CPU
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Генерация случайной матрицы
void generateRandomMatrix(vector<double>& matrix, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-10.0, 10.0);
    
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }
}

// Проверка корректности результатов
bool verifyResult(const vector<double>& C_cpu, 
                  const vector<double>& C_gpu, 
                  int N, 
                  double epsilon = 1e-4) {
    double max_diff = 0.0;
    
    for (int i = 0; i < N * N; ++i) {
        double diff = abs(C_cpu[i] - C_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    if (max_diff > epsilon) {
        cout << "Ошибка верификации! Максимальная разница: " << max_diff << endl;
        return false;
    }
    
    cout << "Верификация пройдена. Максимальная разница: " << max_diff << endl;
    return true;
}

// Измерение времени выполнения
template<typename Func>
double measureTime(Func func, const string& name, int iterations = 1) {
    // Прогрев
    if (iterations > 1) {
        func();
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double avgTime = duration.count() / iterations;
    
    cout << fixed << setprecision(4);
    cout << name << ": " << avgTime << " секунд";
    if (iterations > 1) {
        cout << " (среднее за " << iterations << " запусков)";
    }
    cout << endl;
    
    return avgTime;
}

int main() {
    cout << "\n========================================" << endl;
    cout << "Умножение матриц: CPU vs GPU (CUDA)" << endl;
    cout << "========================================\n" << endl;
    
    // Проверка наличия GPU
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        cout << "GPU не найден!" << endl;
        cout << "Включите GPU в меню: Runtime -> Change runtime type -> Hardware accelerator: GPU" << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "GPU: " << prop.name << endl;
    cout << "Вычислительная способность: " << prop.major << "." << prop.minor << endl;
    cout << "Объем памяти: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << "Количество мультипроцессоров: " << prop.multiProcessorCount << endl;
    
    // Размеры матриц для тестирования
    vector<int> sizes = {100, 250, 500, 750, 1000, 1250, 1500};
    
    cout << "\n" << string(90, '-') << endl;
    cout << left << setw(12) << "Размер"
         << setw(18) << "CPU (сек)"
         << setw(18) << "GPU (сек)"
         << setw(12) << "Ускорение"
         << "Статус" << endl;
    cout << string(90, '-') << endl;
    
    // Хранение результатов для анализа
    vector<int> tested_sizes;
    vector<double> cpu_times, gpu_times, speedups;
    
    for (int N : sizes) {
        cout << "Обработка матриц " << N << "x" << N << "..." << endl;
        
        vector<double> A(N * N);
        vector<double> B(N * N);
        vector<double> C_cpu(N * N);
        vector<double> C_gpu(N * N);
        
        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);
        
        int iterations = (N <= 500) ? 3 : 1;
        
        double cpuTime = measureTime([&]() { multiplyMatricesCPU(A, B, C_cpu, N); }, "CPU", iterations);
        double gpuTime = measureTime([&]() { multiplyMatricesGPU(A, B, C_gpu, N); }, "GPU", iterations);
        
        bool verified = verifyResult(C_cpu, C_gpu, N);
        double speedup = (gpuTime > 0) ? cpuTime / gpuTime : 0.0;
        
        cout << left << setw(12) << N
             << setw(18) << fixed << setprecision(4) << cpuTime
             << setw(18) << setprecision(4) << gpuTime
             << setw(12) << setprecision(2) << speedup
             << (verified ? "УСПЕШНО" : "ОШИБКА") << endl;
        cout << string(90, '-') << endl;
        
        tested_sizes.push_back(N);
        cpu_times.push_back(cpuTime);
        gpu_times.push_back(gpuTime);
        speedups.push_back(speedup);
    }
    
    // Итоговый анализ
    cout << "\n" << string(60, '=') << endl;
    cout << "АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ" << endl;
    cout << string(60, '=') << endl;
    
    double best_speedup = 0;
    double total_speedup = 0;
    for (double s : speedups) {
        if (s > best_speedup) best_speedup = s;
        total_speedup += s;
    }
    double avg_speedup = total_speedup / speedups.size();
    
    cout << "\nМаксимальное ускорение: " << fixed << setprecision(2) << best_speedup << "x" << endl;
    cout << "Среднее ускорение: " << fixed << setprecision(2) << avg_speedup << "x" << endl;
    
    cout << "\nВсе эксперименты завершены!" << endl;
    
    return 0;
}
