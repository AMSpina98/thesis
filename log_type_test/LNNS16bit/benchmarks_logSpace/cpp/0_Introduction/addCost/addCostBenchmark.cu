// ============================================================================
// addCostBenchmark.cu
//
// Benchmark #4 (dei 5 richiesti dal prof): costo della SINGOLA operazione di
// ADDIZIONE, LNNS vs float, al variare del numero di addizioni concatenate K.
//
// Idea: isolare il costo per-operazione (non per-elemento, non per-byte)
// dell'addizione, per spiegare quantitativamente il fattore ~33x di
// istruzioni osservato in matMul (32 accumuli per elemento di output).
//
// Metodologia (in linea con quella già usata per vectorAdd.cu):
//  - warm-up scartato prima delle misure
//  - per ogni K in un set di valori, cudaEvent timing con N campioni,
//    e si salvano media/mediana/stddev/p95
//  - la durata è dominata dal loop di accumulo (K applicazioni di operator+),
//    NON da traffico di memoria: ogni thread pre-carica un piccolo set di
//    operandi (STEPS) in un array locale una sola volta, poi esegue tutto il
//    loop di accumulo su quei valori. Questo evita che il benchmark torni a
//    misurare bandwidth come vectorAdd.
//  - il costo per-operazione si stima poi FUORI da questo programma con una
//    regressione lineare Durata(K) = a + b*K sui dati CSV prodotti: "b" è il
//    costo marginale di UNA addizione, "a" è l'overhead fisso di lancio.
//
// ATTENZIONE (pericolo di ottimizzazione del compilatore):
//  Se si accumula ripetutamente lo STESSO operando costante
//  (acc = acc + step, K volte, con step fisso), per il float il compilatore
//  PUO' teoricamente riconoscere il pattern e trasformarlo; per LNNS invece
//  Log/Exp sono funzioni opache (chiamate SFU), quindi il loop non è
//  algebricamente riducibile a forma chiusa e resta al sicuro. Per estrema
//  sicurezza (e per trattare add/mul in modo identico, vedi mulCostBenchmark)
//  si fa comunque ciclare l'operando su una piccola finestra di STEPS valori
//  distinti pre-caricati, cosa che rende la riduzione algebrica ancora più
//  improbabile da applicare per il compilatore in entrambi i casi.
//
// Da adattare al vostro progetto:
//  - il path di include e l'alias di tipo LNNS_T qui sotto
//  - il modo in cui si costruisce un LNNS da un float (qui si assume un
//    costruttore LNNS_T(float); se nel vostro LNNS_type.cuh si chiama
//    diversamente, es. LNNS_T::FromFloat(x), cambiate solo la riga indicata)
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "../../LNNS_type.cuh"

// Alias del formato LNNS sotto test: cambiate qui per testare 16_10 / 16_11 / 32_21
using LNNS_T = LNNS<uint16_t, 10>;

constexpr int STEPS = 32;   // finestra di operandi distinti (vedi nota sopra)
constexpr int N_ELEMENTS = 1 << 20;   // 1M thread logici, ok per saturare la GPU
constexpr int THREADS_PER_BLOCK = 256;

// ----------------------------------------------------------------------------
// Kernel generico a catena di addizioni, T e' il tipo (LNNS_T o float)
// ----------------------------------------------------------------------------
template <typename T>
__global__ void addChainKernel(const T* __restrict__ src, T* __restrict__ out,
                                int n, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    T local[STEPS];
    #pragma unroll
    for (int s = 0; s < STEPS; s++) {
        local[s] = src[(i + s) % n];
    }

    T acc = local[0];
    for (int k = 0; k < K; k++) {
        acc = acc + local[k % STEPS];
    }

    out[i] = acc;
}

// ----------------------------------------------------------------------------
// Statistiche su un vettore di campioni (stessa logica usata in vectorAdd.cu)
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

// ----------------------------------------------------------------------------
// Esegue il benchmark per un dato K su un dato formato, ritorna le statistiche
// ----------------------------------------------------------------------------
template <typename T>
static Stats runAddBenchmark(const T* d_src, T* d_out, int n, int K,
                              int nWarmup, int nIter,
                              int blocksPerGrid, int threadsPerBlock)
{
    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

    for (int i = 0; i < nWarmup; i++) {
        addChainKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_src, d_out, n, K);
    }
    cudaDeviceSynchronize();

    std::vector<float> samples(nIter);
    for (int i = 0; i < nIter; i++) {
        cudaEventRecord(startE);
        addChainKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_src, d_out, n, K);
        cudaEventRecord(stopE);
        cudaEventSynchronize(stopE);
        cudaEventElapsedTime(&samples[i], startE, stopE);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel error (K=" << K << "): "
                      << cudaGetErrorString(err) << "\n";
        }
    }

    cudaEventDestroy(startE);
    cudaEventDestroy(stopE);
    return computeStats(samples);
}

// ----------------------------------------------------------------------------
// CSV: una riga per (Format, K) con tutte le statistiche in ms
// ----------------------------------------------------------------------------
static void appendResultCSV(const std::string& filename,
                             const std::string& format, int K, const Stats& st)
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
        file << "Format,Operation,K,Mean_ms,Median_ms,StdDev_ms,P95_ms\n";
    }
    file << format << ",Add," << K << ","
         << st.mean << "," << st.median << "," << st.stddev << "," << st.p95 << "\n";
}

int main()
{
    const int n = N_ELEMENTS;
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    const int nWarmup = 200;
    const int nIter   = 2000;   // meno di vectorAdd (300000) perche' qui si
                                // ripete per molti K: 2000 campioni con
                                // mediana/p95 sono comunque stabili

    // K da testare: piccola progressione, coerente con "32 accumuli" osservati in matMul
    std::vector<int> kValues = {1, 2, 4, 8, 16, 32, 64, 128};

    // --- Dati host: valori casuali in un range "sicuro" (lontano da overflow/
    //     cancellazione esatta), per non introdurre rami di saturazione che
    //     sporcherebbero la misura del costo "puro" di operator+ ---
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);

    std::vector<float> h_srcFloat(n);
    for (auto& v : h_srcFloat) v = dist(rng);

    // Conversione a LNNS: ADATTATE questa riga se il vostro costruttore/metodo
    // di conversione da float si chiama diversamente
    std::vector<LNNS_T> h_srcLNNS(n);
    for (int i = 0; i < n; i++) h_srcLNNS[i] = LNNS_T(h_srcFloat[i]);

    // --- Allocazioni device ---
    float   *d_srcFloat, *d_outFloat;
    LNNS_T  *d_srcLNNS,  *d_outLNNS;
    cudaMalloc(&d_srcFloat, n * sizeof(float));
    cudaMalloc(&d_outFloat, n * sizeof(float));
    cudaMalloc(&d_srcLNNS,  n * sizeof(LNNS_T));
    cudaMalloc(&d_outLNNS,  n * sizeof(LNNS_T));

    cudaMemcpy(d_srcFloat, h_srcFloat.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcLNNS,  h_srcLNNS.data(),  n * sizeof(LNNS_T), cudaMemcpyHostToDevice);

    const std::string outCsv = "add_cost_benchmark.csv";

    for (int K : kValues) {
        Stats stFloat = runAddBenchmark<float>(d_srcFloat, d_outFloat, n, K,
                                                nWarmup, nIter, blocksPerGrid, threadsPerBlock);
        appendResultCSV(outCsv, "float", K, stFloat);

        Stats stLNNS = runAddBenchmark<LNNS_T>(d_srcLNNS, d_outLNNS, n, K,
                                                nWarmup, nIter, blocksPerGrid, threadsPerBlock);
        appendResultCSV(outCsv, "LNNS16_10", K, stLNNS);

        std::cout << "K=" << K
                  << "  float median=" << stFloat.median << "ms"
                  << "  LNNS median=" << stLNNS.median << "ms\n";
    }

    cudaFree(d_srcFloat); cudaFree(d_outFloat);
    cudaFree(d_srcLNNS);  cudaFree(d_outLNNS);

    std::cout << "Risultati salvati in " << outCsv << "\n";
    return 0;
}
