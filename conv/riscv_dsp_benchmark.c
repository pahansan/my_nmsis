#include "bench.h"

#include <stdio.h>

BENCH_DECLARE_VAR();

int dummy_riscv_conv_partial_f32(
    const float32_t *pSrcA,
    uint32_t srcALen,
    const float32_t *pSrcB,
    uint32_t srcBLen,
    float32_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{
    return 0;
}

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

void fir_riscv_fir_f32(void)
{
    float32_t firStatef32[TEST_LENGTH_SAMPLES + NUM_TAPS - 1];
    float32_t fir_f32_output[TEST_LENGTH_SAMPLES];

    generate_rand_f32(testInput_f32_50Hz_200Hz, TEST_LENGTH_SAMPLES);

    /* clang-format off */
    riscv_fir_instance_f32 S;
    /* clang-format on */
    riscv_fir_init_f32(&S, NUM_TAPS, firCoeffs32LP, firStatef32, TEST_LENGTH_SAMPLES);
    BENCH_START(riscv_fir_f32);
    riscv_fir_f32(&S, testInput_f32_50Hz_200Hz, fir_f32_output, TEST_LENGTH_SAMPLES);
    BENCH_END(riscv_fir_f32);

    return;
}

void fir_riscv_fir_f64(void)
{
    float64_t firStatef64[TEST_LENGTH_SAMPLES_F64 + NUM_TAPS_F64 - 1];
    float64_t fir_f64_output[TEST_LENGTH_SAMPLES_F64];

    generate_rand_f64(testInput_f64_50Hz_200Hz, TEST_LENGTH_SAMPLES_F64);
    generate_rand_f64(testInput_f64_50Hz_200Hz, TEST_LENGTH_SAMPLES_F64);

    /* clang-format off */
    riscv_fir_instance_f64 S;
    /* clang-format on */
    riscv_fir_init_f64(&S, NUM_TAPS_F64, firCoeffs64LP, firStatef64, TEST_LENGTH_SAMPLES_F64);
    BENCH_START(riscv_fir_f64);
    riscv_fir_f64(&S, testInput_f64_50Hz_200Hz, fir_f64_output, TEST_LENGTH_SAMPLES_F64);
    BENCH_END(riscv_fir_f64);
}

void fir_riscv_fir_q7(void)
{
    q7_t firStateq7[TEST_LENGTH_SAMPLES_Q7 + NUM_TAPS_Q7 - 1];
    q7_t fir_q7_output[TEST_LENGTH_SAMPLES_Q7];

    generate_rand_q7(testInput_q7_50Hz_200Hz, TEST_LENGTH_SAMPLES_Q7);
    riscv_float_to_q7(firCoeffs32LP, firCoeffLP_q7, NUM_TAPS_Q7);

    /* clang-format off */
    riscv_fir_instance_q7 S;
    /* clang-format on */
    riscv_fir_init_q7(&S, NUM_TAPS_Q7, firCoeffLP_q7, firStateq7, TEST_LENGTH_SAMPLES_Q7);
    BENCH_START(riscv_fir_q7);
    riscv_fir_q7(&S, testInput_q7_50Hz_200Hz, fir_q7_output, TEST_LENGTH_SAMPLES_Q7);
    BENCH_END(riscv_fir_q7);

    return;
}

void fir_riscv_fir_q15(void)
{
    q15_t firStateq15[TEST_LENGTH_SAMPLES_Q15 + NUM_TAPS_Q15 - 1];
    q15_t fir_q15_output[TEST_LENGTH_SAMPLES_Q15];

    generate_rand_q15(testInput_q15_50Hz_200Hz, TEST_LENGTH_SAMPLES_Q15);
    riscv_float_to_q15(firCoeffs32LP, firCoeffLP_q15, NUM_TAPS_Q15);

    /* clang-format off */
    riscv_fir_instance_q15 S;
    /* clang-format on */
    riscv_fir_init_q15(&S, NUM_TAPS_Q15, firCoeffLP_q15, firStateq15, TEST_LENGTH_SAMPLES_Q15);
    BENCH_START(riscv_fir_q15);
    riscv_fir_q15(&S, testInput_q15_50Hz_200Hz, fir_q15_output, TEST_LENGTH_SAMPLES_Q15);
    BENCH_END(riscv_fir_q15);

    return;
}

void fir_riscv_fir_q31(void)
{
    q31_t firStateq31[TEST_LENGTH_SAMPLES_Q31 + NUM_TAPS_Q31 - 1];
    q31_t fir_q31_output[TEST_LENGTH_SAMPLES];

    generate_rand_q31(testInput_q31_50Hz_200Hz, TEST_LENGTH_SAMPLES_Q31);
    riscv_float_to_q31(firCoeffs32LP, firCoeffLP_q31, NUM_TAPS_Q31);

    /* clang-format off */
    riscv_fir_instance_q31 S;
    /* clang-format on */
    riscv_fir_init_q31(&S, NUM_TAPS_Q31, firCoeffLP_q31, firStateq31, TEST_LENGTH_SAMPLES_Q31);
    BENCH_START(riscv_fir_q31);
    riscv_fir_q31(&S, testInput_q31_50Hz_200Hz, fir_q31_output, TEST_LENGTH_SAMPLES_Q31);
    BENCH_END(riscv_fir_q31);

    return;
}

void fir_riscv_fir_fast_q15(void)
{
    q15_t firStateq15[TEST_LENGTH_SAMPLES + NUM_TAPS - 1];
    q15_t fir_q15_output[TEST_LENGTH_SAMPLES];

    generate_rand_q15(testInput_q15_50Hz_200Hz, TEST_LENGTH_SAMPLES);
    float32_t firCoeff32LP[NUM_TAPS];
    for (int i = 0; i < NUM_TAPS; i++)
    {
        firCoeff32LP[i] = ((float32_t)rand() / (float32_t)RAND_MAX) * 2 - 1;
    }
    riscv_float_to_q15(firCoeff32LP, firCoeffLP_q15, NUM_TAPS);

    /* clang-format off */
    riscv_fir_instance_q15 S;
    /* clang-format on */
    riscv_fir_init_q15(&S, NUM_TAPS, firCoeffLP_q15, firStateq15, TEST_LENGTH_SAMPLES);
    BENCH_START(riscv_fir_fast_q15);
    riscv_fir_fast_q15(&S, testInput_q15_50Hz_200Hz, fir_q15_output, TEST_LENGTH_SAMPLES);
    BENCH_END(riscv_fir_fast_q15);

    return;
}

void fir_riscv_fir_fast_q31(void)
{
    q31_t firStateq31[TEST_LENGTH_SAMPLES + NUM_TAPS - 1];
    q31_t fir_q31_output[TEST_LENGTH_SAMPLES];

    generate_rand_q31(testInput_q31_50Hz_200Hz, TEST_LENGTH_SAMPLES);

    float32_t firCoeff32LP[NUM_TAPS];
    for (int i = 0; i < NUM_TAPS; i++)
    {
        firCoeff32LP[i] = ((float32_t)rand() / (float32_t)RAND_MAX) * 2 - 1;
    }
    riscv_float_to_q31(firCoeff32LP, firCoeffLP_q31, NUM_TAPS);

    /* clang-format off */
    riscv_fir_instance_q31 S;
    /* clang-format on */
    riscv_fir_init_q31(&S, NUM_TAPS, firCoeffLP_q31, firStateq31, TEST_LENGTH_SAMPLES);
    BENCH_START(riscv_fir_fast_q31);
    riscv_fir_fast_q31(&S, testInput_q31_50Hz_200Hz, fir_q31_output, TEST_LENGTH_SAMPLES);
    BENCH_END(riscv_fir_fast_q31);

    return;
}

int main()
{
    printf("Start partial convolution benchmark test:\n");

    convPartial_riscv_conv_partial_f32();
    convPartial_riscv_conv_partial_q7();
    convPartial_riscv_conv_partial_q15();
    convPartial_riscv_conv_partial_q31();
    convPartial_riscv_conv_partial_fast_q15();
    convPartial_riscv_conv_partial_fast_q31();
    convPartial_riscv_conv_partial_opt_q7();
    convPartial_riscv_conv_partial_opt_q15();
    convPartial_riscv_conv_partial_fast_opt_q15();

    printf("\nStart finite impulse response benchmark test:\n");
    fir_riscv_fir_f32();
    fir_riscv_fir_f64();
    fir_riscv_fir_q7();
    fir_riscv_fir_q15();
    fir_riscv_fir_q31();
    fir_riscv_fir_fast_q15();
    fir_riscv_fir_fast_q31();

    printf("All tests are passed.\n");
}
