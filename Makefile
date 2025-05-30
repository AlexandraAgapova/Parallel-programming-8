NVCC := nvcc
STD := -std=c++17
OPT := -O3

SRC := main.cu
OUT := main

# Include and library paths
INC := \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
  -I/usr/local/cuda/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include

LIB := \
  -L/usr/local/cuda/lib64 \
  -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib

# Linked libraries
LINK := -lboost_program_options -lcublas -lcudart

CUDA_VISIBLE ?= 3

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(STD) $(OPT) $(SRC) -o $(OUT) $(INC) $(LIB) $(LINK)

run: $(OUT)
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE) ./$(OUT) --size 512

clean:
	rm -f $(OUT)

.PHONY: all run clean
