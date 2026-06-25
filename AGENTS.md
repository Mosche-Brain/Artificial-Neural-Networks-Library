# Instructions and informations for AI Agents

## local Environment
* The shell by default use /opt/intel/oneapi/setvars.sh containing environment variables for oneAPI

## Architecture
### Code
* Maintain modular architecture
* Implement fundamental linear algebra and parallel math operations in modules/cun
### Compile configuration
* The used numeric type and used backend must be passed as a commandline arguments in `cmake` command

## Commands
#### Building project with oneAPI and FP16 (for example)
```sh
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCUM_USE_F16=ON -DBUILD_USE_MKL=ON
cmake --build . -j $(nproc --all)
```

# okej