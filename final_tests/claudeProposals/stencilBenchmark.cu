// ============================================================================
// stencilBenchmark.cu
//
// Benchmark custom (scritto da zero, non da samples -- CUDA samples usa
// texture memory per gli stencil, incompatibile con tipi custom come
// LNNS/Encapsulator, che non sono texture-fetchabili).
//
// Stencil 2D a 5 punti (Jacobi iteration / equazione del calore): il
// pattern "da manuale" degli stencil GPU, non una forma inventata --
// e' esattamente lo stencil usato in Rodinia (hotspot), PolyBench
// (jacobi-2d), e nella letteratura standard sugli stencil computation:
//
//   out[i][j] = w_c*in[i][j]
//             + w_n*(in[i-1][j] + in[i+1][j] + in[i][j-1] + in[i][j+1])
//
// con w_c = w_n = 0.2 (5 termini, pesi uguali che sommano a 1 -- forma
// usata in PolyBench jacobi-2d-imper).
//
// Perche' e' un buon 4o/5o benchmark "reale" (diverso da vecAdd/matMul/
// vecMatMul che avete gia'):
//  - Intensita' aritmetica intermedia: 5 mul + 4 add per elemento di
//    output, contro 1 op di vecAdd e 32 di matMul -- un punto in mezzo
//    sulla curva "overhead LNNS cresce con l'intensita' aritmetica".
//  - Accesso a halo in 2 dimensioni (non solo orizzontale come transpose):
//    stressa la shared memory con un tipo custom di dimensione diversa
//    da 4 byte (es. LNNS16_10 e' 2 byte) su un pattern di accesso diverso
//    sia da transpose (puro movimento dati) sia da matMul (accumulo
//    lineare su una riga/colonna).
//  - Bordi: clamp-to-edge (replica del bordo), scelta standard per non
//    introdurre condizioni al contorno arbitrarie nel confronto.
//
// Uso: ./stencilBenchmark <format> <N>
//   format in {float, lnns16_10, lnns16_11, lnns32_21, encaps_float}
//   N = lato della griglia quadrata (default 4096, deve essere multiplo
//       di TILE_DIM)
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

#include "type_dispatch.cuh"

constexpr int RADIUS    = 1;    // stencil a 5 punti: raggio 1 in ciascuna direzione
constexpr int TILE_DIM  = 16;   // tile 2D (16x16 = 256 thread/blocco)

// ----------------------------------------------------------------------------
// Kernel: shared memory con halo di RADIUS su ogni lato del tile 2D.
// Coefficienti passati come T separati (stessa scelta di addCost/mulCost:
// evita di dover indicizzare un array costante di T in modo generico).
// ----------------------------------------------------------------------------
template <typename T>
__global__ void stencil2DKernel(const T* __restrict__ in, T* __restrict__ out,
                                 int n, T wCenter, T wNeighbor)
{
    __shared__ T tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];

    int gx = blockIdx.x * TILE_DIM + threadIdx.x;
    int gy = blockIdx.y * TILE_DIM + threadIdx.y;

    auto clampIdx = [n] __device__ (int idx) {
        if (idx < 0) return 0;
        if (idx >= n) return n - 1;
        return idx;
    };

    int lx = threadIdx.x + RADIUS;
    int ly = threadIdx.y + RADIUS;

    // centro del tile
    tile[ly][lx] = in[clampIdx(gy) * n + clampIdx(gx)];

    // halo: solo i thread sul bordo del tile caricano gli elementi esterni
    if (threadIdx.x < RADIUS) {
        tile[ly][lx - RADIUS]          = in[clampIdx(gy) * n + clampIdx(gx - RADIUS)];
        tile[ly][lx + TILE_DIM]        = in[clampIdx(gy) * n + clampIdx(gx + TILE_DIM)];
    }
    if (threadIdx.y < RADIUS) {
        tile[ly - RADIUS][lx]          = in[clampIdx(gy - RADIUS) * n + clampIdx(gx)];
        tile[ly + TILE_DIM][lx]        = in[clampIdx(gy + TILE_DIM) * n + clampIdx(gx)];
    }

    __syncthreads();

    if (gx < n && gy < n) {
        T acc = wCenter * tile[ly][lx];
        acc = acc + wNeighbor * tile[ly - 1][lx];
        acc = acc + wNeighbor * tile[ly + 1][lx];
        acc = acc + wNeighbor * tile[ly][lx - 1];
        acc = acc + wNeighbor * tile[ly][lx + 1];
        out[gy * n + gx] = acc;
    }
}

// ----------------------------------------------------------------------------
// Statistiche (identiche agli altri benchmark)
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

template <typename T>
static Stats runStencilBenchmark(const T* d_in, T* d_out, int n,
                                  T wCenter, T wNeighbor,
                                  int nWarmup, int nIter)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

    for (int i = 0; i < nWarmup; i++) {
        stencil2DKernel<T><<<grid, block>>>(d_in, d_out, n, wCenter, wNeighbor);
    }
    cudaDeviceSynchronize();

    std::vector<float> samples(nIter);
    for (int i = 0; i < nIter; i++) {
        cudaEventRecord(startE);
        stencil2DKernel<T><<<grid, block>>>(d_in, d_out, n, wCenter, wNeighbor);
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
                             int n, const Stats& st)
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
        file << "Format,N,Mean_ms,Median_ms,StdDev_ms,P95_ms\n";
    }
    file << format << "," << n << ","
         << st.mean << "," << st.median << "," << st.stddev << "," << st.p95 << "\n";
}

// ----------------------------------------------------------------------------
// Runner per il dispatcher condiviso
// ----------------------------------------------------------------------------
struct StencilRunner {
    int n;   // lato della griglia N x N

    template <typename T>
    void operator()(const std::string& formatName)
    {
        size_t total = static_cast<size_t>(n) * n;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.1f, 10.0f);

        std::vector<float> h_inFloat(total);
        for (auto& v : h_inFloat) v = dist(rng);

        std::vector<T> h_in(total);
        for (size_t i = 0; i < total; i++) h_in[i] = T(h_inFloat[i]);

        // Pesi standard PolyBench jacobi-2d: 5 termini uguali che sommano a 1
        // (0.2 ciascuno) -- niente di arbitrario, e' la forma di riferimento.
        T wCenter   = T(0.2f);
        T wNeighbor = T(0.2f);

        T *d_in, *d_out;
        cudaMalloc(&d_in,  total * sizeof(T));
        cudaMalloc(&d_out, total * sizeof(T));
        cudaMemcpy(d_in, h_in.data(), total * sizeof(T), cudaMemcpyHostToDevice);

        const int nWarmup = 20;
        const int nIter   = 200;   // griglia grande: meno iterazioni di add/mulCost

        Stats st = runStencilBenchmark<T>(d_in, d_out, n, wCenter, wNeighbor, nWarmup, nIter);
        appendResultCSV("stencil_benchmark.csv", formatName, n, st);

        std::cout << formatName << "  median=" << st.median << "ms"
                  << "  mean=" << st.mean << "ms\n";

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

    dispatchByTypeName(formatArg, StencilRunner{ n });

    std::cout << "Risultati salvati in stencil_benchmark.csv\n";
    return 0;
}
