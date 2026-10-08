# S3_lut -- addizionatore LNNS a tabella (tabelle + valutatore + tipo LNNS_S3)

Sostituisce la parte float di `LNNS::operator+/-` (Log/Exp/conversioni) con una funzione
dell'unico intero k = mantissa maggiore - mantissa minore, tabulata e interpolata in aritmetica intera.

| File | Ruolo |
|---|---|
| `gen_s3_tables.py` | Generatore parametrico (uno solo per tutti i formati). Simula bit a bit la pipeline intera, verifica TUTTI i k contro il valore ideale, emette l'header. |
| `S3_tables_16_10.cuh` | Generato: LNNS<uint16_t,10>, interpolazione lineare, 8,1 KB (voci (T,dT) da 8 B, M ed EMIN comuni a f+ e f-). |
| `S3_tables_32_21.cuh` | Generato: LNNS<uint32_t,21>, interpolazione quadratica, 24,4 KB (voci da 16 B, M comune, EMIN diverso). |
| `S3_lut_common.cuh` | Macro `S3_HD` e dichiarazione di `s3lut::Tab<F>`. |
| `S3_lut_eval.cuh` | `s3lut::delta<F, PLUS>(k)` e `delta_sel<F>(k, same_sign)`: R+(k) / R-(k). Codice unico host/device. k limitato a K_CUT (voce finale a zero), M comune, una sola lettura vettoriale per voce. |
| `S3.cuh` | Il tipo `LNNS_S3<T,F>` (alias `S3_16_10`, `S3_32_21`): stessa interfaccia di `LNNS`, autonomo (non include S2.cuh). Somma/sottrazione = un solo nucleo intero + `delta_sel`. Conversioni di bordo con log/exp in double. |
| `test_s3_type_host.cpp` + `stub/` | Test host del tipo contro il vero `S2.cuh` (casi speciali, `*`, `/`, confronti identici bit a bit) e contro l'ideale in long double (+, -: errore mantissa <= 0,75 LSB). `stub/cuda_fp16.h` e' vuoto: serve solo a compilare S2.cuh con g++. |
| `test_s3_lut_host.cpp` | Test host (senza GPU): checksum esaustivi identici al generatore + confronto in long double. |

## Rigenerare
    python3 gen_s3_tables.py --preset 16_10
    python3 gen_s3_tables.py --preset 32_21
    python3 gen_s3_tables.py --F 23 --order 2 --g 3 --target 0.5      # altri formati
    (--share-emin / --no-share-emin: stesso EMIN per f+ e f- -> una select in meno, tabella f+ piu' grande)
Servono `numpy` e `mpmath`. Cambiare `--target` (errore massimo prima dell'arrotondamento, in LSB)
sposta il compromesso precisione / dimensione tabella.

## Verificare
    g++ -std=c++17 -O2 -I. test_s3_lut_host.cpp -o test_s3_lut_host && ./test_s3_lut_host

    g++ -std=c++17 -O2 -Wall -Wextra -I. -Istub -I.. test_s3_type_host.cpp -o test_s3_type_host && ./test_s3_type_host
(`-I..` serve a trovare `S2.cuh` nella cartella `cpp`.)

## Uso
    #include "S3_lut/S3.cuh"
    using T = S3_32_21;     // oppure S3_16_10
Le tabelle sono `__device__ const` (cache L1/costante del dispositivo) e copie host `static const` (servono solo al percorso host e ai test).

## Stato
- Verificato su host con g++: checksum bit-exact su tutti i k; tipo contro S2 e contro l'ideale (S3_16_10: errore max 0,53 LSB; S3_32_21: 0,70 LSB).
- Diverso da S2 per costruzione: S2 sbaglia di piu' (float) a |ln x| grande (fino a ~60 LSB nei test); S3 no. Conversioni da/verso double non limitate a e^88.
- NON ancora compilato con nvcc / eseguito su GPU (da fare sul PC: attenzione a `if constexpr`, `__clz`, array `__device__` negli header).
- k=0 nella sottrazione non e' valido per `delta` (gestito a monte in `add_core`: x-x = zero).
