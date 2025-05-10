GCC=/opt/riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-gcc
CLANG=/opt/llvm-project/bin/clang
SPACEMIT_GCC=/opt/spacemit-toolchain/bin/riscv64-unknown-linux-gnu-gcc
SPACEMIT_CLANG=/opt/spacemit-toolchain/bin/clang
CLAGS=-static -O3 -march=rv64gcv
NAME=dsp_benchmark
SRC=riscv_dsp_benchmark.c

.PHONY: all clang gcc spacemit-clang spacemit-gcc

all: clang gcc spacemit-clang spacemit-gcc

clang: $(SRC)
	mkdir -p clang
	$(CLANG) $(CLAGS) $(SRC) -o clang/$(NAME)

gcc: $(SRC)
	mkdir -p gcc
	$(GCC) $(CLAGS) $(SRC) -o gcc/$(NAME)

spacemit-clang:
	mkdir -p spacemit-clang
	$(SPACEMIT_CLANG) $(CLAGS) $(SRC) -o spacemit-clang/$(NAME)

spacemit-gcc:
	mkdir -p spacemit-gcc
	$(SPACEMIT_GCC) $(CLAGS) $(SRC) -o spacemit-gcc/$(NAME)
