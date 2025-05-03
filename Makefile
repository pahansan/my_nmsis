CC=/opt/riscv-gnu-toolchain/bin/riscv64-unknown-linux-gnu-gcc
CLAGS=-static -O3
NAME=dsp_benchmark
SRC=riscv_dsp_benchmark.c

all: $(SRC)
	$(CC) $(CLAGS) $(SRC) -o $(NAME)