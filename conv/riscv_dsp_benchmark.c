#include "bench.h"

#include <stdio.h>

BENCH_DECLARE_VAR();

void convPartial_riscv_conv_partial_f32(void)
{
    float32_t conv_partial_f32_output[2 * max(ARRAYA_SIZE_F32, ARRAYB_SIZE_F32)];

    generate_rand_f32(test_conv_input_f32_A, ARRAYA_SIZE_F32);
    generate_rand_f32(test_conv_input_f32_B, ARRAYB_SIZE_F32);

    BENCH_START(riscv_conv_partial_f32);
    riscv_status result = riscv_conv_partial_f32(test_conv_input_f32_A, ARRAYA_SIZE_F32, test_conv_input_f32_B, ARRAYB_SIZE_F32,
                                                 conv_partial_f32_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_f32);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_q7(void)
{
    q7_t conv_partial_q7_output[2 * max(ARRAYA_SIZE_Q7, ARRAYB_SIZE_Q7)];

    generate_rand_q7(test_conv_input_q7_A, ARRAYA_SIZE_Q7);
    generate_rand_q7(test_conv_input_q7_B, ARRAYB_SIZE_Q7);

    BENCH_START(riscv_conv_partial_q7);
    riscv_status result = riscv_conv_partial_q7(test_conv_input_q7_A, ARRAYA_SIZE_Q7, test_conv_input_q7_B, ARRAYB_SIZE_Q7,
                                                conv_partial_q7_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_q7);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_q15(void)
{
    q15_t conv_partial_q15_output[2 * max(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15)];

    generate_rand_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15);
    generate_rand_q15(test_conv_input_q15_B, ARRAYB_SIZE_Q15);

    BENCH_START(riscv_conv_partial_q15);
    riscv_status result = riscv_conv_partial_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15, test_conv_input_q15_B, ARRAYB_SIZE_Q15,
                                                 conv_partial_q15_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_q15);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_q31(void)
{
    q31_t conv_partial_q31_output[2 * max(ARRAYA_SIZE_Q31, ARRAYB_SIZE_Q31)];

    generate_rand_q31(test_conv_input_q31_A, ARRAYA_SIZE_Q31);
    generate_rand_q31(test_conv_input_q31_B, ARRAYB_SIZE_Q31);

    BENCH_START(riscv_conv_partial_q31);
    riscv_status result = riscv_conv_partial_q31(test_conv_input_q31_A, ARRAYA_SIZE_Q31, test_conv_input_q31_B, ARRAYB_SIZE_Q31,
                                                 conv_partial_q31_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_q31);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_fast_q15(void)
{
    q15_t conv_partial_fast_q15_output[2 * max(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15)];

    generate_rand_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15);
    generate_rand_q15(test_conv_input_q15_B, ARRAYB_SIZE_Q15);

    BENCH_START(riscv_conv_partial_fast_q15);
    riscv_status result = riscv_conv_partial_fast_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15, test_conv_input_q15_B,
                                                      ARRAYB_SIZE_Q15, conv_partial_fast_q15_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_fast_q15);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_fast_q31(void)
{
    q31_t conv_partial_fast_q31_output[2 * max(ARRAYA_SIZE_Q31, ARRAYB_SIZE_Q31)];

    generate_rand_q31(test_conv_input_q31_A, ARRAYA_SIZE_Q31);
    generate_rand_q31(test_conv_input_q31_B, ARRAYB_SIZE_Q31);

    BENCH_START(riscv_conv_partial_fast_q31);
    riscv_status result = riscv_conv_partial_fast_q31(test_conv_input_q31_A, ARRAYA_SIZE_Q31, test_conv_input_q31_B,
                                                      ARRAYB_SIZE_Q31, conv_partial_fast_q31_output, firstIndex, numPoints);
    BENCH_END(riscv_conv_partial_fast_q31);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_opt_q7(void)
{
    q7_t conv_q7_output[2 * max(ARRAYA_SIZE_Q7, ARRAYB_SIZE_Q7)];

    generate_rand_q7(test_conv_input_q7_A, ARRAYA_SIZE_Q7);
    generate_rand_q7(test_conv_input_q7_B, ARRAYB_SIZE_Q7);

    BENCH_START(riscv_conv_partial_opt_q7);
    riscv_status result = riscv_conv_partial_opt_q7(test_conv_input_q7_A, ARRAYA_SIZE_Q7, test_conv_input_q7_B, ARRAYB_SIZE_Q7,
                                                    conv_q7_output, firstIndex, numPoints, pScratch1, pScratch2);
    BENCH_END(riscv_conv_partial_opt_q7);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_opt_q15(void)
{
    q15_t conv_q15_output[2 * max(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15)];

    generate_rand_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15);
    generate_rand_q15(test_conv_input_q15_B, ARRAYB_SIZE_Q15);

    BENCH_START(riscv_conv_partial_opt_q15);
    riscv_status result = riscv_conv_partial_opt_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15, test_conv_input_q15_B, ARRAYB_SIZE_Q15,
                                                     conv_q15_output, firstIndex, numPoints, pScratch1, pScratch2);
    BENCH_END(riscv_conv_partial_opt_q15);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

void convPartial_riscv_conv_partial_fast_opt_q15(void)
{
    q15_t conv_q15_output[2 * max(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15)];

    generate_rand_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15);
    generate_rand_q15(test_conv_input_q15_B, ARRAYB_SIZE_Q15);

    BENCH_START(riscv_conv_partial_fast_opt_q15);
    riscv_status result = riscv_conv_partial_fast_opt_q15(test_conv_input_q15_A, ARRAYA_SIZE_Q15,
                                                          test_conv_input_q15_B, ARRAYB_SIZE_Q15, conv_q15_output, firstIndex,
                                                          numPoints, q15_pScratch1, q15_pScratch2);
    BENCH_END(riscv_conv_partial_fast_opt_q15);

    TEST_ASSERT_EQUAL(RISCV_MATH_SUCCESS, result);

    return;
}

int main()
{
    printf("Start FilteringFunctions benchmark test:\n");

    convPartial_riscv_conv_partial_f32();
    convPartial_riscv_conv_partial_q7();
    convPartial_riscv_conv_partial_q15();
    convPartial_riscv_conv_partial_q31();
    convPartial_riscv_conv_partial_fast_q15();
    convPartial_riscv_conv_partial_fast_q31();
    convPartial_riscv_conv_partial_opt_q7();
    convPartial_riscv_conv_partial_opt_q15();
    convPartial_riscv_conv_partial_fast_opt_q15();

    printf("All tests are passed.\n");
    printf("test for FilteringFunctions benchmark finished.\n");
}
