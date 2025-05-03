#include "bench.h"
#include <stdio.h>

BENCH_DECLARE_VAR();

void rfft_riscv_rfft_f32(void)
{
    float32_t rfft_f32_output[2 * RFFTSIZE];
    riscv_rfft_instance_f32 SS;
    riscv_cfft_radix4_instance_f32 S_CFFT;
    generate_rand_f32(rfft_testinput_f32_50hz_200Hz, RFFTSIZE);

    riscv_status result = riscv_rfft_init_f32(&SS, &S_CFFT, RFFTSIZE, ifftFlag, doBitReverse);
    BENCH_START(riscv_rfft_f32);
    riscv_rfft_f32(&SS, rfft_testinput_f32_50hz_200Hz, rfft_f32_output);
    BENCH_END(riscv_rfft_f32);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);
}

void rfft_riscv_rfft_fast_f32(void)
{
    float32_t rfft_fast_f32_output[2 * RFFTSIZE];

    generate_rand_f32(rfft_testinput_f32_50hz_200Hz_fast, RFFTSIZE);
    riscv_rfft_fast_instance_f32 SS;

    riscv_status result = riscv_rfft_fast_init_f32(&SS, RFFTSIZE);
    BENCH_START(riscv_rfft_fast_f32);
    riscv_rfft_fast_f32(&SS, rfft_testinput_f32_50hz_200Hz_fast, rfft_fast_f32_output, ifftFlag);
    BENCH_END(riscv_rfft_fast_f32);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);
}

int main()
{
    printf("Start TransformFunctions/rfft benchmark test:\n");

    rfft_riscv_rfft_f32();
    // rfft_riscv_rfft_q15();
    // rfft_riscv_rfft_q31();

    // rfft_riscv_rfft_fast_f16();
    rfft_riscv_rfft_fast_f32();

    printf("All tests are passed.\n");
    printf("test for TransformFunctions/rfft benchmark finished.\n");
}
