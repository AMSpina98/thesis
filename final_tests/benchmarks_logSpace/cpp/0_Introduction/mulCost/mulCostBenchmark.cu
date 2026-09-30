// ============================================================================
// mulCostBenchmark.cu
//
// Benchmark #5 (dei 5 richiesti dal prof): costo della SINGOLA operazione di
// MOLTIPLICAZIONE, LNNS vs float, al variare del numero di moltiplicazioni
// concatenate K. Speculare a addCostBenchmark.cu (stessa struttura, stesso
// protocollo di misura), cambia solo operator* al posto di operator+.
//
// Perche' e' interessante: nella LNNS la moltiplicazione e' (teoricamente)
// quasi gratis, somma delle mantisse (i log) + xor dei segni, niente SFU.
// Il confronto Add vs Mul su questo harness gemello dovrebbe quantificare
// esattamente quanto "quasi gratis" sia in pratica, e spiegare (insieme al
// benchmark #4) il fattore ~33x di istruzioni osservato in matMul.
//
// ATTENZIONE - RISCHIO SPECIFICO DI QUESTO BENCHMARK (diverso da quello
// dell'addizione): la moltiplicazione LNNS e' internamente un'ADDIZIONE DI
// INTERI (mantissa_out = mantissa_a + mantissa_b, in un campo fixed-point).
// Una catena "acc = acc * step" ripetuta K volte diventa, a livello di
// mantissa, un accumulo lineare (mantissa_acc += mantissa_step, K volte) —
// esattamente il pattern che LLVM/ptxas sanno riconoscere e "collassare" con
// strength reduction in un singolo mantissa_acc += K * mantissa_step,
// eliminando il loop! Se questo accadesse, misurereste un tempo ~costante
// indipendente da K, e concludereste erroneamente che "la moltiplicazione
// non costa nulla" — non perche' sia vero, ma perche' il compilatore ha
// semplicemente saltato il lavoro.
//
// Mitigazione adottata qui: come in addCostBenchmark, si cicla su una
// finestra di STEPS operandi DISTINTI (non un singolo operando costante), il
// che rende molto piu' difficile per il compilatore riconoscere e collassare
// la sequenza in forma chiusa (dovrebbe rilevare la periodicita' su un K
// runtime, cosa che nvcc/ptxas non fanno in pratica). Ma la mitigazione va
// VERIFICATA, non assunta:
//
//   ==> Dopo aver compilato, controllate con:
//         cuobjdump -sass il_vostro_binario | grep -c <mnemonic istruzione>
//       oppure, meglio, profilate con:
//         ncu --metrics sm__inst_executed.sum --csv ./bench > check.csv
//       per un paio di K (es. K=8 e K=64) e verificate che il numero di
//       istruzioni eseguite scali circa linearmente con K. Se invece resta
//       costante, il loop e' stato ottimizzato via e il benchmark non e'
//       valido: in tal caso bisognerebbe rompere la periodicita' (es. STEPS
//       operandi letti con un indice pseudo-casuale invece che k % STEPS, o
//       una piccola dipendenza dal risultato precedente che non sia pura
//       linearita').
//
// Da adattare al vostro progetto: stesse note di addCostBenchmark.cu
// (path di include, alias LNNS_T, costruttore da float).
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

#include "../../S2.cuh"   // <-- adattate il path se necessario

using LNNS_T = LNNS<uint16_t, 10>;

constexpr int STEPS = 32;
constexpr int N_ELEMENTS = 1 << 20;
constexpr int THREADS_PER_BLOCK = 256;

// ----------------------------------------------------------------------------
// Kernel a catena di moltiplicazioni
// ----------------------------------------------------------------------------
template <typename T>
__global__ void mulChainKernel(const T* __restrict__ src, T* __restrict__ out,
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
        acc = acc * local[k % STEPS];
    }

    out[i] = acc;
}

// ----------------------------------------------------------------------------
// Statistiche (identiche a addCostBenchmark.cu)
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
static Stats runMulBenchmark(const T* d_src, T* d_out, int n, int K,
                              int nWarmup, int nIter,
                              int blocksPerGrid, int threadsPerBlock)
{
    cudaEvent_t startE, stopE;
    cudaEventCreate(&startE);
    cudaEventCreate(&stopE);

    for (int i = 0; i < nWarmup; i++) {
        mulChainKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_src, d_out, n, K);
    }
    cudaDeviceSynchronize();

    std::vector<float> samples(nIter);
    for (int i = 0; i < nIter; i++) {
        cudaEventRecord(startE);
        mulChainKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_src, d_out, n, K);
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
    file << format << ",Mul," << K << ","
         << st.mean << "," << st.median << "," << st.stddev << "," << st.p95 << "\n";
}

int main()
{
    const int n = N_ELEMENTS;
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    const int nWarmup = 200;
    const int nIter   = 2000;

    std::vector<int> kValues = {1, 2, 4, 8, 16, 32, 64, 128};

    // Range operandi vicino a 1.0: per la moltiplicazione, valori troppo
    // lontani da 1 in catene lunghe fanno esplodere/collassare rapidamente
    // il risultato verso +-inf/0 (overflow/underflow di mantissa), il che
    // introdurrebbe i rami di clamp MAX_MANTISSA/MIN_MANTISSA nel conteggio
    // -- esattamente il tipo di confondimento che si vuole evitare qui.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.95f, 1.05f);

    std::vector<float> h_srcFloat(n);
    for (auto& v : h_srcFloat) v = dist(rng);

    std::vector<LNNS_T> h_srcLNNS(n);
    for (int i = 0; i < n; i++) h_srcLNNS[i] = LNNS_T(h_srcFloat[i]);

    float   *d_srcFloat, *d_outFloat;
    LNNS_T  *d_srcLNNS,  *d_outLNNS;
    cudaMalloc(&d_srcFloat, n * sizeof(float));
    cudaMalloc(&d_outFloat, n * sizeof(float));
    cudaMalloc(&d_srcLNNS,  n * sizeof(LNNS_T));
    cudaMalloc(&d_outLNNS,  n * sizeof(LNNS_T));

    cudaMemcpy(d_srcFloat, h_srcFloat.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcLNNS,  h_srcLNNS.data(),  n * sizeof(LNNS_T), cudaMemcpyHostToDevice);

    const std::string outCsv = "mul_cost_benchmark.csv";

    for (int K : kValues) {
        Stats stFloat = runMulBenchmark<float>(d_srcFloat, d_outFloat, n, K,
                                                nWarmup, nIter, blocksPerGrid, threadsPerBlock);
        appendResultCSV(outCsv, "float", K, stFloat);

        Stats stLNNS = runMulBenchmark<LNNS_T>(d_srcLNNS, d_outLNNS, n, K,
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
