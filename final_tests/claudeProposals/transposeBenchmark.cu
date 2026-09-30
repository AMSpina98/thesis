// ============================================================================
// transposeBenchmark.cu
//
// Benchmark "opzionale" (6/6 se lo portate anche voi al prof): trasposizione
// di una matrice quadrata NxN, float vs LNNS vs Encapsulator, usando il
// dispatcher condiviso in type_dispatch.cuh.
//
// Perche' e' interessante nonostante sia quasi tutto movimento dati:
//  - Zero (o quasi zero) aritmetica: nessuna add/mul, solo un load e uno
//    store con indici scambiati. Se qui LNNS ed Encapsulator restano vicini
//    a float (a parte l'overhead di pack/unpack per i formati storage < 32
//    bit), e' la controprova che l'overhead osservato altrove (matMul,
//    add/mulCostBenchmark) e' davvero dominato dal COSTO delle operazioni
//    aritmetiche via SFU, non da un costo "di base" del tipo custom.
//  - Pattern di accesso non coalescente sul lato naive (out[x][y] = in[y][x]):
//    utile anche come sanity check che i tipi custom non introducano
//    ulteriori penalita' di accesso oltre a quella intrinseca del transpose.
//
// Due kernel:
//  - transposeNaive: read coalescente, write strided (o viceversa a seconda
//    di come la si legge) -- baseline didattica.
//  - transposeTiled: tile in shared memory + swap degli indici in shared,
//    per avere sia read che write coalescenti. E' la versione "seria" da
//    riportare come numero principale; la naive resta come confronto.
//
// Uso: ./transposeBenchmark <format> <N>
//   format in {float, lnns16_10, lnns16_11, lnns32_21, encaps_float}
//   N = lato della matrice (default 4096, deve essere multiplo di TILE_DIM)
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <random>

#include "type_dispatch.cuh"   // porta con se' LNNS_type.cuh ed encapsulator.cuh

constexpr int TILE_DIM   = 32;
constexpr int BLOCK_ROWS = 8;   // ogni thread copre TILE_DIM/BLOCK_ROWS righe

// ----------------------------------------------------------------------------
// Kernel naive: nessuna shared memory, uno dei due lati (qui la write) non
// e' coalescente.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void transposeNaive(const T* __restrict__ in, T* __restrict__ out, int n)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[x * n + (y + j)] = in[(y + j) * n + x];
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel tiled: carica un tile in shared memory (read coalescente), poi lo
// riscrive trasposto (write coalescente), leggendo dalla shared con gli
// indici scambiati. +1 di padding sulla colonna per evitare bank conflict.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void transposeTiled(const T* __restrict__ in, T* __restrict__ out, int n)
{
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
        }
    }
    __syncthreads();

    // blocco trasposto: swap dei blocchi di indici, non degli indici dentro
    // il thread
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ----------------------------------------------------------------------------
// Statistiche (stessa logica di add/mulCostBenchmark)
// ----------------------------------------------------------------------------
struct Stats { float mean, median, stddev, p95; };

static Stats computeStats(std::vector<float> samples)
{
    Stats st{};
    size_t n = samples.size();
    double sum = 0.0;
    for (float v : samples) sum += v;
    st.mean = static_cast<float>(sum / n);

    double sqSum = 0.0;
    for (float v : samples) sqSum += (double(v) - st.mean) * (double(v) - st.mean);
    st.stddev = static_cast<float>(std::sqrt(sqSum / n));

    std::sort(samples.begin(), samples.end());
    st.median = samples[n / 2];
    st.p95    = samples[static_cast<size_t>(0.95 * n)];
    return st;
}

enum class Kernel { Naive, Tiled };

template <typename T>
static Stats runTransposeBenchmark(const T* d_in, T* d_out, int n, Kernel kind,
                                    int nWarmup, int nIter)
{
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(n / TILE_DIM, n / TILE_DIM);

    auto launch = [&]() {
        if (kind == Kernel::Naive) transposeNaive<T><<<grid, block>>>(d_in, d_out, n);
        else                       transposeTiled<T><<<grid, block>>>(d_in, d_out, n);
    };

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

    for (int i = 0; i < nWarmup; i++) launch();
    cudaDeviceSynchronize();

    std::vector<float> samples(nIter);
    for (int i = 0; i < nIter; i++) {
        cudaEventRecord(startE);
        launch();
        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        cudaEventElapsedTime(&samples[i], startE, stopE);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        }
    }

    cudaEventDestroy(startE);
    cudaEventDestroy(stopE);
    return computeStats(samples);
}

static void appendResultCSV(const std::string& filename, const std::string& format,
                             const std::string& kernelName, int n, const Stats& st)
{
    std::ifstream check(filename);
    bool writeHeader = !check.good();
    check.close();

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore apertura " << filename << "\n";
        return;
    }
    if (writeHeader) {
        file << "Format,Kernel,N,Mean_ms,Median_ms,StdDev_ms,P95_ms\n";
    }
    file << format << "," << kernelName << "," << n << ","
         << st.mean << "," << st.median << "," << st.stddev << "," << st.p95 << "\n";
}

// ----------------------------------------------------------------------------
// Runner: la logica specifica di QUESTO benchmark, T viene forni to a runtime
// dal dispatcher in base all'argomento da riga di comando.
// ----------------------------------------------------------------------------
struct TransposeRunner {
    int n;

    template <typename T>
    void operator()(const std::string& formatName)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.1f, 10.0f);

        std::vector<float> h_inFloat(static_cast<size_t>(n) * n);
        for (auto& v : h_inFloat) v = dist(rng);

        std::vector<T> h_in(static_cast<size_t>(n) * n);
        for (size_t i = 0; i < h_in.size(); i++) h_in[i] = T(h_inFloat[i]);

        T *d_in, *d_out;
        cudaMalloc(&d_in,  h_in.size() * sizeof(T));
        cudaMalloc(&d_out, h_in.size() * sizeof(T));
        cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(T), cudaMemcpyHostToDevice);

        const int nWarmup = 20;
        const int nIter   = 200;   // matrici grandi: meno iterazioni di add/mulCost

        const std::string outCsv = "transpose_benchmark.csv";

        Stats stNaive = runTransposeBenchmark<T>(d_in, d_out, n, Kernel::Naive, nWarmup, nIter);
        appendResultCSV(outCsv, formatName, "naive", n, stNaive);

        Stats stTiled = runTransposeBenchmark<T>(d_in, d_out, n, Kernel::Tiled, nWarmup, nIter);
        appendResultCSV(outCsv, formatName, "tiled", n, stTiled);

        std::cout << formatName
                  << "  naive median=" << stNaive.median << "ms"
                  << "  tiled median=" << stTiled.median << "ms\n";

        cudaFree(d_in);
        cudaFree(d_out);
    }
};

int main(int argc, char** argv)
{
    std::string formatArg = (argc > 1) ? argv[1] : "float";
    int n = (argc > 2) ? std::atoi(argv[2]) : 4096;

    if (n % TILE_DIM != 0) {
        std::cerr << "N deve essere multiplo di TILE_DIM (" << TILE_DIM << ")\n";
        return 1;
    }

    dispatchByTypeName(formatArg, TransposeRunner{ n });

    std::cout << "Risultati salvati in transpose_benchmark.csv\n";
    return 0;
}
