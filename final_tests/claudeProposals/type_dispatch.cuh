#pragma once

// ============================================================================
// type_dispatch.cuh — l'if/else "quale tipo LNNS/float/Encapsulator uso"
// scritto UNA sola volta, condiviso da tutti i benchmark.
//
// Ogni benchmark NON riscrive questa lista: include questo header e passa
// un proprio "runner" (una struct con un operator() templato) che contiene
// solo la logica specifica di QUEL benchmark. La lista di tipi vive qui,
// in un solo posto -- se aggiungete/togliete una variante, si cambia
// solo questo file, non i 5 benchmark.
//
// Uso da un benchmark:
//
//   struct MyRunner {
//       int n;   // eventuali parametri che vi servono
//       template <typename T>
//       void operator()(const std::string& formatName) {
//           runBenchmark<T>(formatName, n);   // la vostra logica esistente
//       }
//   };
//
//   int main(int argc, char** argv) {
//       std::string typeArg = (argc > 1) ? argv[1] : "float";
//       dispatchByTypeName(typeArg, MyRunner{ /* n = */ 1<<20 });
//   }
// ============================================================================

#include <string>
#include <stdexcept>

#include "LNNS_type.cuh"
#include "encapsulator.cuh"

// Adattate qui gli alias alle varianti che usate davvero -- e' l'UNICO posto
// dove vanno elencate.
using LNNS16_10 = LNNS<unsigned short, 10>;
using LNNS16_11 = LNNS<unsigned short, 11>;
using LNNS32_21 = LNNS<unsigned int, 21>;

template <typename Callable>
void dispatchByTypeName(const std::string& name, Callable&& callable) {
    if (name == "float") {
        callable.template operator()<float>(name);
    } else if (name == "lnns16_10") {
        callable.template operator()<LNNS16_10>(name);
    } else if (name == "lnns16_11") {
        callable.template operator()<LNNS16_11>(name);
    } else if (name == "lnns32_21") {
        callable.template operator()<LNNS32_21>(name);
    } else if (name == "encaps_float") {
        callable.template operator()<Encapsulator<float>>(name);
    } else {
        throw std::runtime_error("Formato non riconosciuto: '" + name +
            "' (attesi: float, lnns16_10, lnns16_11, lnns32_21, encaps_float)");
    }
}
