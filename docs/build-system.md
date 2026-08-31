# Przegląd Systemu Kompilacji Projektu YANN

## 1. Podsumowanie systemu

**Projekt:** YANN (Yet Another Neural Network)  
**System kompilacji:** CMake (3.10+ w głównym pliku, 3.14 w modułach)  
**Język:** C++23  
**Kompilator:** Intel oneAPI (icpx/icx)  

## 2. Struktura CMake

```
YANN/
├── CMakeLists.txt              # Główny plik konfiguracyjny
├── src/CMakeLists.txt          # Główna biblioteka YANN
├── modules/cum/
│   ├── CMakeLists.txt          # Moduł obliczeń numerycznych
│   └── sources/CMakeLists.txt  # Backendy (MKL, CBLAS, Native)
├── examples/CMakeLists.txt     # Przykłady
└── bindings/C/CMakeLists.txt   # C API
```

## 3. Opcje kompilacji

### Opcje budowania
| Opcja | Domyślnie | Opis |
|-------|-----------|------|
| `BUILD_SHARED_LIBS` | ON | Buduj jako biblioteka dzielona |
| `BUILD_EXAMPLES` | ON | Buduj przykłady |
| `BUILD_PYTHON_BINDINGS` | OFF | Python bindings (pybind11) |
| `BUILD_C_API` | ON | Stabilne C API |
| `ENABLE_DEBUG_OUTPUT` | ON | Logi debugowania |

### Backendy matematyczne
| Opcja | Domyślnie | Opis |
|-------|-----------|------|
| `BUILD_USE_EIGEN` | OFF | Eigen |
| `BUILD_USE_OPENBLAS` | OFF | OpenBLAS |
| `BUILD_USE_MKL` | **ON** | Intel oneMKL (domyślny) |
| `BUILD_USE_NATIVE` | OFF | Natywna implementacja C++ |

### Precyzja obliczeń
| Opcja | Domyślnie | Opis |
|-------|-----------|------|
| `CUM_USE_F64` | OFF | Podwójna precyzja (64-bit) |
| `CUM_USE_F32` | OFF | Pojedyncza precyzja (32-bit) |
| `CUM_USE_F16` | OFF | Półprecyzja (16-bit) |
| `CUM_USE_BF16` | OFF | Brain float (16-bit) |
| `CUM_USE_Q8` | OFF | Kwantyzacja (8-bit int) |

**Zachowanie domyślne:** Jeśli żadna precyzja nie jest wybrana, używa F32 z ostrzeżeniem.

## 4. Zidentyfikowane problemy

### 4.1. Zakodowane na stałe ścieżki do Intel oneAPI

**Lokalizacje:**
- `modules/cum/CMakeLists.txt:37` — `/opt/intel/oneapi/compiler/latest/include/`
- `modules/cum/sources/CMakeLists.txt:14` — ścieżki do TBB i kompilatora
- `modules/cum/sources/Backends/MKL/CMakeLists.txt:11-12,19` — ścieżki do TBB i MKL
- `examples/CMakeLists.txt` — brak, ale używa `-march=native`

**Problem:** Brak przenośności — projekt wymaga specyficznej instalacji Intel oneAPI w `/opt/intel/oneapi/`

**Rekomendacja:** Użyć zmiennych środowiskowych lub Find* modules:

```cmake
find_path(INTEL_COMPILER_DIR include/sycl
        PATHS $ENV{CMAKE_CXX_COMPILER}
        PATH_SUFFIXES ../..)
```

### 4.2. Powtarzające się opcje kompilatora

**Wzorzec powtórzeń:**
- `src/CMakeLists.txt:25-34`
- `examples/CMakeLists.txt:34-41`
- `bindings/C/CMakeLists.txt:73-79`
- `modules/cum/sources/Backends/MKL/CMakeLists.txt:30-32`

**Opcje powtórzone:**
```cmake
-qopenmp           # OpenMP
-march=native      # Architektura (problemy z przenośnością!)
-Wall              # Ostrzeżenia
-O3                # Optymalizacja
-ffast-math        # Szybka matematyka
-mavxvnni          # Rozszerzenia AVX (specyficzne dla Intel)
-fsycl             # SYCL (tylko dla MKL)
```

**Rekomendacja:** Utworzyć plik `cmake/CompileOptions.cmake` z wspólnymi opcjami:
```cmake
# cmake/CompileOptions.cmake
option(ENABLE_NATIVE_ARCH "Use -march=native (disables cross-compilation)" OFF)
set(COMMON_FLAGS -Wall -O3 -ffast-math -qopenmp)
if(ENABLE_NATIVE_ARCH)
    list(APPEND COMMON_FLAGS -march=native -mavxvnni)
endif()
```

### 4.3. Wymuszanie kompilatora

**Lokalizacje:**
- `modules/cum/CMakeLists.txt:7-8`
- `modules/cum/sources/Backends/MKL/CMakeLists.txt:4-5`

```cmake
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)
```

**Problem:** Nadpisuje kompilator ustawiony przez użytkownika lub system, ignoruje `-DCMAKE_CXX_COMPILER=`

**Rekomendacja:** Usunąć te linie lub uczynić opcjonalnymi:
```cmake
option(FORCE_INTEL_COMPILER "Force Intel compiler (use only with Intel oneAPI)" OFF)
if(FORCE_INTEL_COMPILER)
    find_program(INTEL_CXX icpx)
    if(INTEL_CXX)
        set(CMAKE_CXX_COMPILER ${INTEL_CXX})
    endif()
endif()
```

### 4.4. Opcje `-march=native` i `-mavxvnni`

**Lokalizacje:** Wszystkie pliki CMakeLists.txt

**Problem:** 
- `-march=native` — tworzy binarki niezgodne z innymi procesorami
- `-mavxvnni` — specyficzne dla CPU Intel z VNNI, nie działa na AMD

**Rekomendacja:** Domyślnie wyłączyć, dodać opcję:
```cmake
option(ENABLE_NATIVE_OPTIMIZATIONS "Enable CPU-specific optimizations (-march=native -mavxvnni)" OFF)
```

### 4.5. Brakujące sprawdzenie błędów

**Lokalizacja:** `modules/cum/sources/Backends/MKL/CMakeLists.txt:11-12`

```cmake
find_package(TBB CONFIG REQUIRED PATHS ...)
find_package(MKL CONFIG REQUIRED PATHS ...)
```

**Problem:** Jeśli MKL nie zostanie znaleziony, błąd może być niejasny. Brak jasnych komunikatów jak zainstalować zależności.

### 4.6. Wyłączone reguły instalacyjne

**Lokalizacja:** `src/CMakeLists.txt:45-64` (zakomentowane)

```cmake
# install(TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     LIBRARY DESTINATION lib
#     ...
# )
```

**Problem:** Brak możliści instalacji projektu (brak `make install`)

### 4.7. Niespójne użycie GLOB

**src/CMakeLists.txt:1** — poprawnie używa `CONFIGURE_DEPENDS`:
```cmake
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "*.cpp")
```

**modules/cum/sources/Backends/MKL/CMakeLists.txt:2** — bez `CONFIGURE_DEPENDS`:
```cmake
file(GLOB_RECURSE SOURCES_MKL *.cpp *.c)
```

**Problem:** Bez `CONFIGURE_DEPENDS` CMake nie wykrywa nowych plików bez ręcznego czyszczenia i przebudowy.

### 4.8. Literówki w definicjach precyzji

**Lokalizacja:** `CMakeLists.txt:87-100`

```cmake
list(APPEND YANN_COMPILE_DEFINITIONS CUM_CUM_USE_F32)  # Podwójne CUM_
list(APPEND YANN_COMPILE_DEFINITIONS CUM_CUM_USE_F16)  # Podwójne CUM_
list(APPEND YANN_COMPILE_DEFINITIONS CUM_CUM_USE_BF16) # Podwójne CUM_
list(APPEND YANN_COMPILE_DEFINITIONS CUM_CUM_USE_Q8)   # Podwójne CUM_
```

**Problem:** Literówki — `CUM_CUM_` zamiast `CUM_`

## 5. Analiza zależności

```
YANN (main library)
├── OpenMP
├── nlohmann_json
└── cum::Core
    └── cum::MKL (optional)
        ├── Intel TBB
        └── Intel MKL
```

## 6. Dobra praktyka (pozytywne wzorce)

1. **Modułowa architektura** — podział na src, modules, examples, bindings
2. **Walidacja wyboru dokładności** — sprawdzenie czy wybrano dokładnie jedną opcję precyzji
3. **Generator expressions** — użycie `$<BUILD_INTERFACE:...>` dla install
4. **Aliasy bibliotek** — `add_library(cum::Core ALIAS ...)` ułatwia linkowanie
5. **Osobne moduły cum** — czytelny podział na Core i backendy MKL/CBLAS/Native

## 7. Rekomendacje priorytetowe

| Priorytet | Problem | Akcja |
|-----------|---------|-------|
| **Wysoki** | Zakodowane ścieżki | Użyć ENV variables lub find_package |
| **Wysoki** | Opcje `-march=native` | Domyślnie wyłączyć |
| **Wysoki** | Wymuszanie kompilatora | Usunąć lub opcjonalizować |
| **Średni** | Powtórzenia opcji kompilacji | Utworzyć wspólny plik cmake |
| **Średni** | Wyłączone install rules | Włączyć lub usunąć komentarze |
| **Niski** | Literówki CUM_CUM_ | Poprawić na CUM_ |
| **Niski** | Niespójne GLOB | Dodać CONFIGURE_DEPENDS |


## 8. Naprawione problemy

### 8.1. Literówki CUM_CUM_ → CUM_ (NAPRAWIONO)
Poprawiono w `CMakeLists.txt` — definicje precyzji miały podwójne `CUM_`.

### 8.2. Definicje precyzji nie docierały do modułu cum (NAPRAWIONO)
Dodano `add_compile_definitions()` przed `add_subdirectory(modules/cum)` — wcześniej moduł cum nie otrzymywał definicji `CUM_USE_F16` itd., co powodowało błędy linkowania.

### 8.3. Zakodowane defines w przykładach (NAPRAWIONO)
Usunięto `#define CUM_USE_F16` i `#define BUILD_USE_MKL` z plików przykładów — teraz precyzja jest podawana wyłącznie przez argument cmake.

## 9. Weryfikacja kompilacji

Przetestowano kompilację z opcjami:
```bash
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCUM_USE_F16=ON -DBUILD_USE_MKL=ON
```

**Wynik:** ✅ Sukces
- `playground` — skompilowany, działa poprawnie (rozmiar cumeric_t = 2 bajty)
- `tests` — skompilowany

## 10. Przykładowy przykład kompilacji

```bash
# Kompilacja z domyślnymi ustawieniami (MKL + F32)
mkdir build && cd build
cmake ..
cmake --build . -j $(nproc)

# Kompilacja z F16
cmake .. -DCUM_USE_F16=ON -DBUILD_USE_MKL=ON

# Kompilacja bez MKL (native)
cmake .. -DBUILD_USE_MKL=OFF -DBUILD_USE_NATIVE=ON
```

## 11. Wnioski

System kompilacji YANN jest funkcjonalny, ale ma problemy z przenośnością i utrzymaniem. Główne problemy to:

1. **Silne powiązanie z Intel oneAPI** — zakodowane ścieżki i wymuszanie kompilatora
2. **Powtórzenia kodu** — te same opcje kompilatora w 4+ miejscach
3. **Problemy z przenośnością** — `-march=native` nie pozwala na dystrybucję binarną

Projekt nadaje się do użytku wewnętrznego na maszynach z Intel oneAPI, ale wymaga refaktoryzacji przed szerszą dystrybucją.