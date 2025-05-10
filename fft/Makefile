GCC=/opt/riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-gcc
CLANG=/opt/llvm-project/bin/clang
CLAGS=-static -O3
NAME=dsp_benchmark
SRC=riscv_dsp_benchmark.c

.PHONY: all clang gcc

all: clang gcc

clang: $(SRC)
	mkdir -p clang
	$(CLANG) $(CLAGS) $(SRC) -o clang/$(NAME)

gcc: $(SRC)
	mkdir -p gcc
	$(GCC) $(CLAGS) $(SRC) -o gcc/$(NAME)
