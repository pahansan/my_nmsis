#pragma once

#include "real_coef.h"

#include <stdint.h>
#include <stdlib.h>

typedef float float32_t;

#define BENCH_DECLARE_VAR() static volatile uint64_t _bc_sttcyc, _bc_endcyc, _bc_usecyc, _bc_sumcyc, _bc_lpcnt, _bc_ercd;

#define RFFTSIZE (512)

static uint8_t ifftFlag = 0;
static uint8_t doBitReverse = 1;
static float32_t rfft_testinput_f32_50hz_200Hz[RFFTSIZE];
static float32_t rfft_testinput_f32_50hz_200Hz_fast[RFFTSIZE];

typedef struct
{
    uint16_t fftLen;              /**< length of the FFT. */
    uint8_t ifftFlag;             /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;       /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const float32_t *pTwiddle;    /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable; /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;    /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;        /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
    float32_t onebyfftLen;        /**< value of 1/fftLen. */
} riscv_cfft_radix4_instance_f32;

typedef struct
{
    uint32_t fftLenReal;                   /**< length of the real FFT. */
    uint16_t fftLenBy2;                    /**< length of the complex FFT. */
    uint8_t ifftFlagR;                     /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
    uint8_t bitReverseFlagR;               /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
    uint32_t twidCoefRModifier;            /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    const float32_t *pTwiddleAReal;        /**< points to the real twiddle factor table. */
    const float32_t *pTwiddleBReal;        /**< points to the imag twiddle factor table. */
    riscv_cfft_radix4_instance_f32 *pCfft; /**< points to the complex FFT instance. */
} riscv_rfft_instance_f32;

typedef enum
{
    RISCV_MATH_SUCCESS = 0,               /**< No error */
    RISCV_MATH_ARGUMENT_ERROR = -1,       /**< One or more arguments are incorrect */
    RISCV_MATH_LENGTH_ERROR = -2,         /**< Length of data buffer is incorrect */
    RISCV_MATH_SIZE_MISMATCH = -3,        /**< Size of matrices is not compatible with the operation */
    RISCV_MATH_NANINF = -4,               /**< Not-a-number (NaN) or infinity is generated */
    RISCV_MATH_SINGULAR = -5,             /**< Input matrix is singular and cannot be inverted */
    RISCV_MATH_TEST_FAILURE = -6,         /**< Test Failed */
    RISCV_MATH_DECOMPOSITION_FAILURE = -7 /**< Decomposition Failed */
} riscv_status;

#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#endif

typedef unsigned long rv_csr_t;

#ifndef __ASM
#define __ASM __asm
#endif

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

#define __RV_CSR_READ(csr)                                                   \
    ({                                                                       \
        rv_csr_t __v;                                                        \
        __ASM volatile("csrr %0, " STRINGIFY(csr) : "=r"(__v) : : "memory"); \
        __v;                                                                 \
    })

// #define CSR_MCYCLE 0xb00
// #define CSR_MCYCLE 0xc00
#define CSR_MCYCLE 0xC01

__STATIC_FORCEINLINE uint64_t __get_rv_cycle(void)
{
#if __RISCV_XLEN == 32
    volatile uint32_t high0, low, high;
    uint64_t full;

    high0 = __RV_CSR_READ(CSR_MCYCLEH);
    low = __RV_CSR_READ(CSR_MCYCLE);
    high = __RV_CSR_READ(CSR_MCYCLEH);
    if (high0 != high)
    {
        low = __RV_CSR_READ(CSR_MCYCLE);
    }
    full = (((uint64_t)high) << 32) | low;
    return full;
#elif __RISCV_XLEN == 64
    return (uint64_t)__RV_CSR_READ(CSR_MCYCLE);
#else // TODO Need cover for XLEN=128 case in future
    return (uint64_t)__RV_CSR_READ(CSR_MCYCLE);
#endif
}

#ifndef READ_CYCLE
/** Read run cycle of cpu */
#define READ_CYCLE __get_rv_cycle
#endif

/** Start to do benchmark for proc, and record start cycle, and reset error code */
#define BENCH_START(proc) \
    _bc_ercd = 0;         \
    _bc_sttcyc = READ_CYCLE();

/** Sample a benchmark for proc, and record this start -> sample cost cycle, and accumulate it to sum cycle */
#define BENCH_SAMPLE(proc)                \
    _bc_endcyc = READ_CYCLE();            \
    _bc_usecyc = _bc_endcyc - _bc_sttcyc; \
    _bc_sumcyc += _bc_usecyc;             \
    _bc_lpcnt += 1;

/** Mark end of benchmark for proc, and calc used cycle, and print it */
#define BENCH_END(proc) \
    BENCH_SAMPLE(proc); \
    printf("CSV, %s, %lu\n", #proc, (unsigned long)_bc_usecyc);

#define TEST_ASSERT_EQUAL(expected, actual)                                                                   \
    do                                                                                                        \
    {                                                                                                         \
        if ((expected) != (actual))                                                                           \
        {                                                                                                     \
            printf("Assertion failed: %s != %s, file %s, line %d\n", #expected, #actual, __FILE__, __LINE__); \
        }                                                                                                     \
    } while (0)

static inline void do_srand(void)
{
    unsigned long randvar = __RV_CSR_READ(CSR_MCYCLE);
    srand(randvar);
    // printf("srandvar is %d\n", randvar);
}

typedef int32_t q31_t;
#define Q31_MAX ((q31_t)(0x7FFFFFFFL))

static void generate_rand_f32(float32_t *src, int length)
{
    do_srand();
    for (int i = 0; i < length; i++)
    {
        src[i] = (float32_t)((rand() % Q31_MAX - Q31_MAX / 2) * 1.0 / Q31_MAX);
    }
}

#ifndef RISCV_DSP_ATTRIBUTE
#define RISCV_DSP_ATTRIBUTE
#endif

// extern const float32_t twiddleCoef_4096[8192];
#define twiddleCoef twiddleCoef_4096

RISCV_DSP_ATTRIBUTE riscv_status riscv_cfft_radix4_init_f32(
    riscv_cfft_radix4_instance_f32 *S,
    uint16_t fftLen,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag)
{
    /*  Initialise the default riscv status */
    riscv_status status = RISCV_MATH_ARGUMENT_ERROR;

    /*  Initialise the default riscv status */
    status = RISCV_MATH_SUCCESS;

    /*  Initialise the FFT length */
    S->fftLen = fftLen;

    /*  Initialise the Twiddle coefficient pointer */
    S->pTwiddle = (float32_t *)twiddleCoef;

    /*  Initialise the Flag for selection of CFFT or CIFFT */
    S->ifftFlag = ifftFlag;

    /*  Initialise the Flag for calculation Bit reversal or not */
    S->bitReverseFlag = bitReverseFlag;

    /*  Initializations of structure parameters depending on the FFT length */
    switch (S->fftLen)
    {

    case 4096U:
        /*  Initializations of structure parameters for 4096 point FFT */

        /*  Initialise the twiddle coef modifier value */
        S->twidCoefModifier = 1U;
        /*  Initialise the bit reversal table modifier */
        S->bitRevFactor = 1U;
        /*  Initialise the bit reversal table pointer */
        S->pBitRevTable = (uint16_t *)riscvBitRevTable;
        /*  Initialise the 1/fftLen Value */
        S->onebyfftLen = 0.000244140625;
        break;

    case 1024U:
        /*  Initializations of structure parameters for 1024 point FFT */

        /*  Initialise the twiddle coef modifier value */
        S->twidCoefModifier = 4U;
        /*  Initialise the bit reversal table modifier */
        S->bitRevFactor = 4U;
        /*  Initialise the bit reversal table pointer */
        S->pBitRevTable = (uint16_t *)&riscvBitRevTable[3];
        /*  Initialise the 1/fftLen Value */
        S->onebyfftLen = 0.0009765625f;
        break;

    case 256U:
        /*  Initializations of structure parameters for 256 point FFT */
        S->twidCoefModifier = 16U;
        S->bitRevFactor = 16U;
        S->pBitRevTable = (uint16_t *)&riscvBitRevTable[15];
        S->onebyfftLen = 0.00390625f;
        break;

    case 64U:
        /*  Initializations of structure parameters for 64 point FFT */
        S->twidCoefModifier = 64U;
        S->bitRevFactor = 64U;
        S->pBitRevTable = (uint16_t *)&riscvBitRevTable[63];
        S->onebyfftLen = 0.015625f;
        break;

    case 16U:
        /*  Initializations of structure parameters for 16 point FFT */
        S->twidCoefModifier = 256U;
        S->bitRevFactor = 256U;
        S->pBitRevTable = (uint16_t *)&riscvBitRevTable[255];
        S->onebyfftLen = 0.0625f;
        break;

    default:
        /*  Reporting argument error if fftSize is not valid value */
        status = RISCV_MATH_ARGUMENT_ERROR;
        break;
    }

    return (status);
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_init_f32(
    riscv_rfft_instance_f32 *S,
    riscv_cfft_radix4_instance_f32 *S_CFFT,
    uint32_t fftLenReal,
    uint32_t ifftFlagR,
    uint32_t bitReverseFlag)
{
    /*  Initialise the default riscv status */
    riscv_status status = RISCV_MATH_ARGUMENT_ERROR;

    /*  Initialise the default riscv status */
    status = RISCV_MATH_SUCCESS;

    /*  Initialize the Real FFT length */
    S->fftLenReal = (uint16_t)fftLenReal;

    /*  Initialize the Complex FFT length */
    S->fftLenBy2 = (uint16_t)fftLenReal / 2U;

    /*  Initialize the Twiddle coefficientA pointer */
    S->pTwiddleAReal = (float32_t *)realCoefA;

    /*  Initialize the Twiddle coefficientB pointer */
    S->pTwiddleBReal = (float32_t *)realCoefB;

    /*  Initialize the Flag for selection of RFFT or RIFFT */
    S->ifftFlagR = (uint8_t)ifftFlagR;

    /*  Initialize the Flag for calculation Bit reversal or not */
    S->bitReverseFlagR = (uint8_t)bitReverseFlag;

    /*  Initializations of structure parameters depending on the FFT length */
    switch (S->fftLenReal)
    {
        /* Init table modifier value */
    case 8192U:
        S->twidCoefRModifier = 1U;
        break;
    case 2048U:
        S->twidCoefRModifier = 4U;
        break;
    case 512U:
        S->twidCoefRModifier = 16U;
        break;
    case 128U:
        S->twidCoefRModifier = 64U;
        break;
    default:
        /*  Reporting argument error if rfftSize is not valid value */
        status = RISCV_MATH_ARGUMENT_ERROR;
        break;
    }

    /* Init Complex FFT Instance */
    S->pCfft = S_CFFT;

    if (S->ifftFlagR)
    {
        /* Initializes the CIFFT Module for fftLenreal/2 length */
        riscv_cfft_radix4_init_f32(S->pCfft, S->fftLenBy2, 1U, 0U);
    }
    else
    {
        /* Initializes the CFFT Module for fftLenreal/2 length */
        riscv_cfft_radix4_init_f32(S->pCfft, S->fftLenBy2, 0U, 0U);
    }

    /* return the status of RFFT Init function */
    return (status);
}

RISCV_DSP_ATTRIBUTE void riscv_split_rifft_f32(
    float32_t *pSrc,
    uint32_t fftLen,
    const float32_t *pATable,
    const float32_t *pBTable,
    float32_t *pDst,
    uint32_t modifier)
{
    float32_t outR, outI;             /* Temporary variables for output */
    const float32_t *pCoefA, *pCoefB; /* Temporary pointers for twiddle factors */
    float32_t CoefA1, CoefA2, CoefB1; /* Temporary variables for twiddle coefficients */
    float32_t *pSrc1 = &pSrc[0], *pSrc2 = &pSrc[(2U * fftLen) + 1U];

    pCoefA = &pATable[0];
    pCoefB = &pBTable[0];

    while (fftLen > 0U)
    {
        /*
          outR = (  pIn[2 * i]             * pATable[2 * i]
                  + pIn[2 * i + 1]         * pATable[2 * i + 1]
                  + pIn[2 * n - 2 * i]     * pBTable[2 * i]
                  - pIn[2 * n - 2 * i + 1] * pBTable[2 * i + 1]);

          outI = (  pIn[2 * i + 1]         * pATable[2 * i]
                  - pIn[2 * i]             * pATable[2 * i + 1]
                  - pIn[2 * n - 2 * i]     * pBTable[2 * i + 1]
                  - pIn[2 * n - 2 * i + 1] * pBTable[2 * i]);
         */

        CoefA1 = *pCoefA++;
        CoefA2 = *pCoefA;

        /* outR = (pSrc[2 * i] * CoefA1 */
        outR = *pSrc1 * CoefA1;

        /* - pSrc[2 * i] * CoefA2 */
        outI = -(*pSrc1++) * CoefA2;

        /* (pSrc[2 * i + 1] + pSrc[2 * fftLen - 2 * i + 1]) * CoefA2 */
        outR += (*pSrc1 + *pSrc2) * CoefA2;

        /* pSrc[2 * i + 1] * CoefA1 */
        outI += (*pSrc1++) * CoefA1;

        CoefB1 = *pCoefB;

        /* - pSrc[2 * fftLen - 2 * i + 1] * CoefB1 */
        outI -= *pSrc2-- * CoefB1;

        /* pSrc[2 * fftLen - 2 * i] * CoefB1 */
        outR += *pSrc2 * CoefB1;

        /* pSrc[2 * fftLen - 2 * i] * CoefA2 */
        outI += *pSrc2-- * CoefA2;

        /* write output */
        *pDst++ = outR;
        *pDst++ = outI;

        /* update coefficient pointer */
        pCoefB = pCoefB + (modifier * 2);
        pCoefA = pCoefA + (modifier * 2 - 1);

        /* Decrement loop count */
        fftLen--;
    }
}

RISCV_DSP_ATTRIBUTE void riscv_radix4_butterfly_inverse_f32(
    float32_t *pSrc,
    uint16_t fftLen,
    const float32_t *pCoef,
    uint16_t twidCoefModifier,
    float32_t onebyfftLen)
{
    float32_t co1, co2, co3, si1, si2, si3;
    uint32_t ia1, ia2, ia3;
    uint32_t i0, i1, i2, i3;
    uint32_t n1, n2, j, k;

#if defined(RISCV_MATH_LOOPUNROLL)

    float32_t xaIn, yaIn, xbIn, ybIn, xcIn, ycIn, xdIn, ydIn;
    float32_t Xaplusc, Xbplusd, Yaplusc, Ybplusd, Xaminusc, Xbminusd, Yaminusc,
        Ybminusd;
    float32_t Xb12C_out, Yb12C_out, Xc12C_out, Yc12C_out, Xd12C_out, Yd12C_out;
    float32_t Xb12_out, Yb12_out, Xc12_out, Yc12_out, Xd12_out, Yd12_out;
    float32_t *ptr1;
    float32_t p0, p1, p2, p3, p4, p5, p6, p7;
    float32_t a0, a1, a2, a3, a4, a5, a6, a7;

    /*  Initializations for the first stage */
    n2 = fftLen;
    n1 = n2;

    /* n2 = fftLen/4 */
    n2 >>= 2U;
    i0 = 0U;
    ia1 = 0U;

    j = n2;

    /*  Calculation of first stage */
    do
    {
        /*  index calculation for the input as, */
        /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
        i1 = i0 + n2;
        i2 = i1 + n2;
        i3 = i2 + n2;

        /*  Butterfly implementation */
        xaIn = pSrc[(2U * i0)];
        yaIn = pSrc[(2U * i0) + 1U];

        xcIn = pSrc[(2U * i2)];
        ycIn = pSrc[(2U * i2) + 1U];

        xbIn = pSrc[(2U * i1)];
        ybIn = pSrc[(2U * i1) + 1U];

        xdIn = pSrc[(2U * i3)];
        ydIn = pSrc[(2U * i3) + 1U];

        /* xa + xc */
        Xaplusc = xaIn + xcIn;
        /* xb + xd */
        Xbplusd = xbIn + xdIn;
        /* ya + yc */
        Yaplusc = yaIn + ycIn;
        /* yb + yd */
        Ybplusd = ybIn + ydIn;

        /*  index calculation for the coefficients */
        ia2 = ia1 + ia1;
        co2 = pCoef[ia2 * 2U];
        si2 = pCoef[(ia2 * 2U) + 1U];

        /* xa - xc */
        Xaminusc = xaIn - xcIn;
        /* xb - xd */
        Xbminusd = xbIn - xdIn;
        /* ya - yc */
        Yaminusc = yaIn - ycIn;
        /* yb - yd */
        Ybminusd = ybIn - ydIn;

        /* xa' = xa + xb + xc + xd */
        pSrc[(2U * i0)] = Xaplusc + Xbplusd;

        /* ya' = ya + yb + yc + yd */
        pSrc[(2U * i0) + 1U] = Yaplusc + Ybplusd;

        /* (xa - xc) - (yb - yd) */
        Xb12C_out = (Xaminusc - Ybminusd);
        /* (ya - yc) + (xb - xd) */
        Yb12C_out = (Yaminusc + Xbminusd);
        /* (xa + xc) - (xb + xd) */
        Xc12C_out = (Xaplusc - Xbplusd);
        /* (ya + yc) - (yb + yd) */
        Yc12C_out = (Yaplusc - Ybplusd);
        /* (xa - xc) + (yb - yd) */
        Xd12C_out = (Xaminusc + Ybminusd);
        /* (ya - yc) - (xb - xd) */
        Yd12C_out = (Yaminusc - Xbminusd);

        co1 = pCoef[ia1 * 2U];
        si1 = pCoef[(ia1 * 2U) + 1U];

        /*  index calculation for the coefficients */
        ia3 = ia2 + ia1;
        co3 = pCoef[ia3 * 2U];
        si3 = pCoef[(ia3 * 2U) + 1U];

        Xb12_out = Xb12C_out * co1;
        Yb12_out = Yb12C_out * co1;
        Xc12_out = Xc12C_out * co2;
        Yc12_out = Yc12C_out * co2;
        Xd12_out = Xd12C_out * co3;
        Yd12_out = Yd12C_out * co3;

        /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
        // Xb12_out -= Yb12C_out * si1;
        p0 = Yb12C_out * si1;
        /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
        // Yb12_out += Xb12C_out * si1;
        p1 = Xb12C_out * si1;
        /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
        // Xc12_out -= Yc12C_out * si2;
        p2 = Yc12C_out * si2;
        /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
        // Yc12_out += Xc12C_out * si2;
        p3 = Xc12C_out * si2;
        /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
        // Xd12_out -= Yd12C_out * si3;
        p4 = Yd12C_out * si3;
        /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
        // Yd12_out += Xd12C_out * si3;
        p5 = Xd12C_out * si3;

        Xb12_out -= p0;
        Yb12_out += p1;
        Xc12_out -= p2;
        Yc12_out += p3;
        Xd12_out -= p4;
        Yd12_out += p5;

        /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
        pSrc[2U * i1] = Xc12_out;

        /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
        pSrc[(2U * i1) + 1U] = Yc12_out;

        /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
        pSrc[2U * i2] = Xb12_out;

        /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
        pSrc[(2U * i2) + 1U] = Yb12_out;

        /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
        pSrc[2U * i3] = Xd12_out;

        /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
        pSrc[(2U * i3) + 1U] = Yd12_out;

        /*  Twiddle coefficients index modifier */
        ia1 = ia1 + twidCoefModifier;

        /*  Updating input index */
        i0 = i0 + 1U;

    } while (--j);

    twidCoefModifier <<= 2U;

    /*  Calculation of second stage to excluding last stage */
    for (k = fftLen >> 2U; k > 4U; k >>= 2U)
    {
        /*  Initializations for the first stage */
        n1 = n2;
        n2 >>= 2U;
        ia1 = 0U;

        /*  Calculation of first stage */
        j = 0;
        do
        {
            /*  index calculation for the coefficients */
            ia2 = ia1 + ia1;
            ia3 = ia2 + ia1;
            co1 = pCoef[ia1 * 2U];
            si1 = pCoef[(ia1 * 2U) + 1U];
            co2 = pCoef[ia2 * 2U];
            si2 = pCoef[(ia2 * 2U) + 1U];
            co3 = pCoef[ia3 * 2U];
            si3 = pCoef[(ia3 * 2U) + 1U];

            /*  Twiddle coefficients index modifier */
            ia1 = ia1 + twidCoefModifier;

            i0 = j;
            do
            {
                /*  index calculation for the input as, */
                /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
                i1 = i0 + n2;
                i2 = i1 + n2;
                i3 = i2 + n2;

                xaIn = pSrc[(2U * i0)];
                yaIn = pSrc[(2U * i0) + 1U];

                xbIn = pSrc[(2U * i1)];
                ybIn = pSrc[(2U * i1) + 1U];

                xcIn = pSrc[(2U * i2)];
                ycIn = pSrc[(2U * i2) + 1U];

                xdIn = pSrc[(2U * i3)];
                ydIn = pSrc[(2U * i3) + 1U];

                /* xa - xc */
                Xaminusc = xaIn - xcIn;
                /* (xb - xd) */
                Xbminusd = xbIn - xdIn;
                /* ya - yc */
                Yaminusc = yaIn - ycIn;
                /* (yb - yd) */
                Ybminusd = ybIn - ydIn;

                /* xa + xc */
                Xaplusc = xaIn + xcIn;
                /* xb + xd */
                Xbplusd = xbIn + xdIn;
                /* ya + yc */
                Yaplusc = yaIn + ycIn;
                /* yb + yd */
                Ybplusd = ybIn + ydIn;

                /* (xa - xc) - (yb - yd) */
                Xb12C_out = (Xaminusc - Ybminusd);
                /* (ya - yc) +  (xb - xd) */
                Yb12C_out = (Yaminusc + Xbminusd);
                /* xa + xc -(xb + xd) */
                Xc12C_out = (Xaplusc - Xbplusd);
                /* (ya + yc) - (yb + yd) */
                Yc12C_out = (Yaplusc - Ybplusd);
                /* (xa - xc) + (yb - yd) */
                Xd12C_out = (Xaminusc + Ybminusd);
                /* (ya - yc) -  (xb - xd) */
                Yd12C_out = (Yaminusc - Xbminusd);

                pSrc[(2U * i0)] = Xaplusc + Xbplusd;
                pSrc[(2U * i0) + 1U] = Yaplusc + Ybplusd;

                Xb12_out = Xb12C_out * co1;
                Yb12_out = Yb12C_out * co1;
                Xc12_out = Xc12C_out * co2;
                Yc12_out = Yc12C_out * co2;
                Xd12_out = Xd12C_out * co3;
                Yd12_out = Yd12C_out * co3;

                /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
                // Xb12_out -= Yb12C_out * si1;
                p0 = Yb12C_out * si1;
                /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
                // Yb12_out += Xb12C_out * si1;
                p1 = Xb12C_out * si1;
                /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
                // Xc12_out -= Yc12C_out * si2;
                p2 = Yc12C_out * si2;
                /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
                // Yc12_out += Xc12C_out * si2;
                p3 = Xc12C_out * si2;
                /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
                // Xd12_out -= Yd12C_out * si3;
                p4 = Yd12C_out * si3;
                /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
                // Yd12_out += Xd12C_out * si3;
                p5 = Xd12C_out * si3;

                Xb12_out -= p0;
                Yb12_out += p1;
                Xc12_out -= p2;
                Yc12_out += p3;
                Xd12_out -= p4;
                Yd12_out += p5;

                /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
                pSrc[2U * i1] = Xc12_out;

                /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
                pSrc[(2U * i1) + 1U] = Yc12_out;

                /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
                pSrc[2U * i2] = Xb12_out;

                /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
                pSrc[(2U * i2) + 1U] = Yb12_out;

                /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
                pSrc[2U * i3] = Xd12_out;

                /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
                pSrc[(2U * i3) + 1U] = Yd12_out;

                i0 += n1;
            } while (i0 < fftLen);
            j++;
        } while (j <= (n2 - 1U));
        twidCoefModifier <<= 2U;
    }
    /*  Initializations of last stage */

    j = fftLen >> 2;
    ptr1 = &pSrc[0];

    /*  Calculations of last stage */
    do
    {
        xaIn = ptr1[0];
        yaIn = ptr1[1];
        xbIn = ptr1[2];
        ybIn = ptr1[3];
        xcIn = ptr1[4];
        ycIn = ptr1[5];
        xdIn = ptr1[6];
        ydIn = ptr1[7];

        /*  Butterfly implementation */
        /* xa + xc */
        Xaplusc = xaIn + xcIn;

        /* xa - xc */
        Xaminusc = xaIn - xcIn;

        /* ya + yc */
        Yaplusc = yaIn + ycIn;

        /* ya - yc */
        Yaminusc = yaIn - ycIn;

        /* xb + xd */
        Xbplusd = xbIn + xdIn;

        /* yb + yd */
        Ybplusd = ybIn + ydIn;

        /* (xb-xd) */
        Xbminusd = xbIn - xdIn;

        /* (yb-yd) */
        Ybminusd = ybIn - ydIn;

        /* xa' = (xa+xb+xc+xd) * onebyfftLen */
        a0 = (Xaplusc + Xbplusd);
        /* ya' = (ya+yb+yc+yd) * onebyfftLen */
        a1 = (Yaplusc + Ybplusd);
        /* xc' = (xa-xb+xc-xd) * onebyfftLen */
        a2 = (Xaplusc - Xbplusd);
        /* yc' = (ya-yb+yc-yd) * onebyfftLen  */
        a3 = (Yaplusc - Ybplusd);
        /* xb' = (xa-yb-xc+yd) * onebyfftLen */
        a4 = (Xaminusc - Ybminusd);
        /* yb' = (ya+xb-yc-xd) * onebyfftLen */
        a5 = (Yaminusc + Xbminusd);
        /* xd' = (xa-yb-xc+yd) * onebyfftLen */
        a6 = (Xaminusc + Ybminusd);
        /* yd' = (ya-xb-yc+xd) * onebyfftLen */
        a7 = (Yaminusc - Xbminusd);

        p0 = a0 * onebyfftLen;
        p1 = a1 * onebyfftLen;
        p2 = a2 * onebyfftLen;
        p3 = a3 * onebyfftLen;
        p4 = a4 * onebyfftLen;
        p5 = a5 * onebyfftLen;
        p6 = a6 * onebyfftLen;
        p7 = a7 * onebyfftLen;

        /* xa' = (xa+xb+xc+xd) * onebyfftLen */
        ptr1[0] = p0;
        /* ya' = (ya+yb+yc+yd) * onebyfftLen */
        ptr1[1] = p1;
        /* xc' = (xa-xb+xc-xd) * onebyfftLen */
        ptr1[2] = p2;
        /* yc' = (ya-yb+yc-yd) * onebyfftLen  */
        ptr1[3] = p3;
        /* xb' = (xa-yb-xc+yd) * onebyfftLen */
        ptr1[4] = p4;
        /* yb' = (ya+xb-yc-xd) * onebyfftLen */
        ptr1[5] = p5;
        /* xd' = (xa-yb-xc+yd) * onebyfftLen */
        ptr1[6] = p6;
        /* yd' = (ya-xb-yc+xd) * onebyfftLen */
        ptr1[7] = p7;

        /* increment source pointer by 8 for next calculations */
        ptr1 = ptr1 + 8U;

    } while (--j);

#else

    float32_t t1, t2, r1, r2, s1, s2;

    /*  Initializations for the first stage */
    n2 = fftLen;
    n1 = n2;

    /*  Calculation of first stage */
    for (k = fftLen; k > 4U; k >>= 2U)
    {
        /*  Initializations for the first stage */
        n1 = n2;
        n2 >>= 2U;
        ia1 = 0U;

        /*  Calculation of first stage */
        j = 0;
        do
        {
            /*  index calculation for the coefficients */
            ia2 = ia1 + ia1;
            ia3 = ia2 + ia1;
            co1 = pCoef[ia1 * 2U];
            si1 = pCoef[(ia1 * 2U) + 1U];
            co2 = pCoef[ia2 * 2U];
            si2 = pCoef[(ia2 * 2U) + 1U];
            co3 = pCoef[ia3 * 2U];
            si3 = pCoef[(ia3 * 2U) + 1U];

            /*  Twiddle coefficients index modifier */
            ia1 = ia1 + twidCoefModifier;

            i0 = j;
            do
            {
                /*  index calculation for the input as, */
                /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
                i1 = i0 + n2;
                i2 = i1 + n2;
                i3 = i2 + n2;

                /* xa + xc */
                r1 = pSrc[(2U * i0)] + pSrc[(2U * i2)];

                /* xa - xc */
                r2 = pSrc[(2U * i0)] - pSrc[(2U * i2)];

                /* ya + yc */
                s1 = pSrc[(2U * i0) + 1U] + pSrc[(2U * i2) + 1U];

                /* ya - yc */
                s2 = pSrc[(2U * i0) + 1U] - pSrc[(2U * i2) + 1U];

                /* xb + xd */
                t1 = pSrc[2U * i1] + pSrc[2U * i3];

                /* xa' = xa + xb + xc + xd */
                pSrc[2U * i0] = r1 + t1;

                /* xa + xc -(xb + xd) */
                r1 = r1 - t1;

                /* yb + yd */
                t2 = pSrc[(2U * i1) + 1U] + pSrc[(2U * i3) + 1U];

                /* ya' = ya + yb + yc + yd */
                pSrc[(2U * i0) + 1U] = s1 + t2;

                /* (ya + yc) - (yb + yd) */
                s1 = s1 - t2;

                /* (yb - yd) */
                t1 = pSrc[(2U * i1) + 1U] - pSrc[(2U * i3) + 1U];

                /* (xb - xd) */
                t2 = pSrc[2U * i1] - pSrc[2U * i3];

                /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
                pSrc[2U * i1] = (r1 * co2) - (s1 * si2);

                /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
                pSrc[(2U * i1) + 1U] = (s1 * co2) + (r1 * si2);

                /* (xa - xc) - (yb - yd) */
                r1 = r2 - t1;

                /* (xa - xc) + (yb - yd) */
                r2 = r2 + t1;

                /* (ya - yc) +  (xb - xd) */
                s1 = s2 + t2;

                /* (ya - yc) -  (xb - xd) */
                s2 = s2 - t2;

                /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
                pSrc[2U * i2] = (r1 * co1) - (s1 * si1);

                /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
                pSrc[(2U * i2) + 1U] = (s1 * co1) + (r1 * si1);

                /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
                pSrc[2U * i3] = (r2 * co3) - (s2 * si3);

                /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
                pSrc[(2U * i3) + 1U] = (s2 * co3) + (r2 * si3);

                i0 += n1;
            } while (i0 < fftLen);
            j++;
        } while (j <= (n2 - 1U));
        twidCoefModifier <<= 2U;
    }
    /*  Initializations of last stage */
    n1 = n2;
    n2 >>= 2U;

    /*  Calculations of last stage */
    for (i0 = 0U; i0 <= (fftLen - n1); i0 += n1)
    {
        /*  index calculation for the input as, */
        /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
        i1 = i0 + n2;
        i2 = i1 + n2;
        i3 = i2 + n2;

        /*  Butterfly implementation */
        /* xa + xc */
        r1 = pSrc[2U * i0] + pSrc[2U * i2];

        /* xa - xc */
        r2 = pSrc[2U * i0] - pSrc[2U * i2];

        /* ya + yc */
        s1 = pSrc[(2U * i0) + 1U] + pSrc[(2U * i2) + 1U];

        /* ya - yc */
        s2 = pSrc[(2U * i0) + 1U] - pSrc[(2U * i2) + 1U];

        /* xc + xd */
        t1 = pSrc[2U * i1] + pSrc[2U * i3];

        /* xa' = xa + xb + xc + xd */
        pSrc[2U * i0] = (r1 + t1) * onebyfftLen;

        /* (xa + xb) - (xc + xd) */
        r1 = r1 - t1;

        /* yb + yd */
        t2 = pSrc[(2U * i1) + 1U] + pSrc[(2U * i3) + 1U];

        /* ya' = ya + yb + yc + yd */
        pSrc[(2U * i0) + 1U] = (s1 + t2) * onebyfftLen;

        /* (ya + yc) - (yb + yd) */
        s1 = s1 - t2;

        /* (yb-yd) */
        t1 = pSrc[(2U * i1) + 1U] - pSrc[(2U * i3) + 1U];

        /* (xb-xd) */
        t2 = pSrc[2U * i1] - pSrc[2U * i3];

        /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
        pSrc[2U * i1] = r1 * onebyfftLen;

        /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
        pSrc[(2U * i1) + 1U] = s1 * onebyfftLen;

        /* (xa - xc) - (yb-yd) */
        r1 = r2 - t1;

        /* (xa - xc) + (yb-yd) */
        r2 = r2 + t1;

        /* (ya - yc) + (xb-xd) */
        s1 = s2 + t2;

        /* (ya - yc) - (xb-xd) */
        s2 = s2 - t2;

        /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
        pSrc[2U * i2] = r1 * onebyfftLen;

        /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
        pSrc[(2U * i2) + 1U] = s1 * onebyfftLen;

        /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
        pSrc[2U * i3] = r2 * onebyfftLen;

        /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
        pSrc[(2U * i3) + 1U] = s2 * onebyfftLen;
    }

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */
}

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_f32(
    float32_t *pSrc,
    uint16_t fftSize,
    uint16_t bitRevFactor,
    const uint16_t *pBitRevTab)
{
    uint16_t fftLenBy2, fftLenBy2p1;
    uint16_t i, j;
    float32_t in;

    /*  Initializations */
    j = 0U;
    fftLenBy2 = fftSize >> 1U;
    fftLenBy2p1 = (fftSize >> 1U) + 1U;

    /* Bit Reversal Implementation */
    for (i = 0U; i <= (fftLenBy2 - 2U); i += 2U)
    {
        if (i < j)
        {
            /*  pSrc[i] <-> pSrc[j]; */
            in = pSrc[2U * i];
            pSrc[2U * i] = pSrc[2U * j];
            pSrc[2U * j] = in;

            /*  pSrc[i+1U] <-> pSrc[j+1U] */
            in = pSrc[(2U * i) + 1U];
            pSrc[(2U * i) + 1U] = pSrc[(2U * j) + 1U];
            pSrc[(2U * j) + 1U] = in;

            /*  pSrc[i+fftLenBy2p1] <-> pSrc[j+fftLenBy2p1] */
            in = pSrc[2U * (i + fftLenBy2p1)];
            pSrc[2U * (i + fftLenBy2p1)] = pSrc[2U * (j + fftLenBy2p1)];
            pSrc[2U * (j + fftLenBy2p1)] = in;

            /*  pSrc[i+fftLenBy2p1+1U] <-> pSrc[j+fftLenBy2p1+1U] */
            in = pSrc[(2U * (i + fftLenBy2p1)) + 1U];
            pSrc[(2U * (i + fftLenBy2p1)) + 1U] =
                pSrc[(2U * (j + fftLenBy2p1)) + 1U];
            pSrc[(2U * (j + fftLenBy2p1)) + 1U] = in;
        }

        /*  pSrc[i+1U] <-> pSrc[j+1U] */
        in = pSrc[2U * (i + 1U)];
        pSrc[2U * (i + 1U)] = pSrc[2U * (j + fftLenBy2)];
        pSrc[2U * (j + fftLenBy2)] = in;

        /*  pSrc[i+2U] <-> pSrc[j+2U] */
        in = pSrc[(2U * (i + 1U)) + 1U];
        pSrc[(2U * (i + 1U)) + 1U] = pSrc[(2U * (j + fftLenBy2)) + 1U];
        pSrc[(2U * (j + fftLenBy2)) + 1U] = in;

        /*  Reading the index for the bit reversal */
        j = *pBitRevTab;

        /*  Updating the bit reversal index depending on the fft length  */
        pBitRevTab += bitRevFactor;
    }
}

RISCV_DSP_ATTRIBUTE void riscv_radix4_butterfly_f32(
    float32_t *pSrc,
    uint16_t fftLen,
    const float32_t *pCoef,
    uint16_t twidCoefModifier)
{
    float32_t co1, co2, co3, si1, si2, si3;
    uint32_t ia1, ia2, ia3;
    uint32_t i0, i1, i2, i3;
    uint32_t n1, n2, j, k;

#if defined(RISCV_MATH_LOOPUNROLL)

    float32_t xaIn, yaIn, xbIn, ybIn, xcIn, ycIn, xdIn, ydIn;
    float32_t Xaplusc, Xbplusd, Yaplusc, Ybplusd, Xaminusc, Xbminusd, Yaminusc,
        Ybminusd;
    float32_t Xb12C_out, Yb12C_out, Xc12C_out, Yc12C_out, Xd12C_out, Yd12C_out;
    float32_t Xb12_out, Yb12_out, Xc12_out, Yc12_out, Xd12_out, Yd12_out;
    float32_t *ptr1;
    float32_t p0, p1, p2, p3, p4, p5;
    float32_t a0, a1, a2, a3, a4, a5, a6, a7;

    /*  Initializations for the first stage */
    n2 = fftLen;
    n1 = n2;

    /* n2 = fftLen/4 */
    n2 >>= 2U;
    i0 = 0U;
    ia1 = 0U;

    j = n2;

    /*  Calculation of first stage */
    do
    {
        /*  index calculation for the input as, */
        /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
        i1 = i0 + n2;
        i2 = i1 + n2;
        i3 = i2 + n2;

        xaIn = pSrc[(2U * i0)];
        yaIn = pSrc[(2U * i0) + 1U];

        xbIn = pSrc[(2U * i1)];
        ybIn = pSrc[(2U * i1) + 1U];

        xcIn = pSrc[(2U * i2)];
        ycIn = pSrc[(2U * i2) + 1U];

        xdIn = pSrc[(2U * i3)];
        ydIn = pSrc[(2U * i3) + 1U];

        /* xa + xc */
        Xaplusc = xaIn + xcIn;
        /* xb + xd */
        Xbplusd = xbIn + xdIn;
        /* ya + yc */
        Yaplusc = yaIn + ycIn;
        /* yb + yd */
        Ybplusd = ybIn + ydIn;

        /*  index calculation for the coefficients */
        ia2 = ia1 + ia1;
        co2 = pCoef[ia2 * 2U];
        si2 = pCoef[(ia2 * 2U) + 1U];

        /* xa - xc */
        Xaminusc = xaIn - xcIn;
        /* xb - xd */
        Xbminusd = xbIn - xdIn;
        /* ya - yc */
        Yaminusc = yaIn - ycIn;
        /* yb - yd */
        Ybminusd = ybIn - ydIn;

        /* xa' = xa + xb + xc + xd */
        pSrc[(2U * i0)] = Xaplusc + Xbplusd;
        /* ya' = ya + yb + yc + yd */
        pSrc[(2U * i0) + 1U] = Yaplusc + Ybplusd;

        /* (xa - xc) + (yb - yd) */
        Xb12C_out = (Xaminusc + Ybminusd);
        /* (ya - yc) + (xb - xd) */
        Yb12C_out = (Yaminusc - Xbminusd);
        /* (xa + xc) - (xb + xd) */
        Xc12C_out = (Xaplusc - Xbplusd);
        /* (ya + yc) - (yb + yd) */
        Yc12C_out = (Yaplusc - Ybplusd);
        /* (xa - xc) - (yb - yd) */
        Xd12C_out = (Xaminusc - Ybminusd);
        /* (ya - yc) + (xb - xd) */
        Yd12C_out = (Xbminusd + Yaminusc);

        co1 = pCoef[ia1 * 2U];
        si1 = pCoef[(ia1 * 2U) + 1U];

        /*  index calculation for the coefficients */
        ia3 = ia2 + ia1;
        co3 = pCoef[ia3 * 2U];
        si3 = pCoef[(ia3 * 2U) + 1U];

        Xb12_out = Xb12C_out * co1;
        Yb12_out = Yb12C_out * co1;
        Xc12_out = Xc12C_out * co2;
        Yc12_out = Yc12C_out * co2;
        Xd12_out = Xd12C_out * co3;
        Yd12_out = Yd12C_out * co3;

        /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
        // Xb12_out -= Yb12C_out * si1;
        p0 = Yb12C_out * si1;
        /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
        // Yb12_out += Xb12C_out * si1;
        p1 = Xb12C_out * si1;
        /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
        // Xc12_out -= Yc12C_out * si2;
        p2 = Yc12C_out * si2;
        /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
        // Yc12_out += Xc12C_out * si2;
        p3 = Xc12C_out * si2;
        /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
        // Xd12_out -= Yd12C_out * si3;
        p4 = Yd12C_out * si3;
        /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
        // Yd12_out += Xd12C_out * si3;
        p5 = Xd12C_out * si3;

        Xb12_out += p0;
        Yb12_out -= p1;
        Xc12_out += p2;
        Yc12_out -= p3;
        Xd12_out += p4;
        Yd12_out -= p5;

        /* xc' = (xa-xb+xc-xd)co2 + (ya-yb+yc-yd)(si2) */
        pSrc[2U * i1] = Xc12_out;

        /* yc' = (ya-yb+yc-yd)co2 - (xa-xb+xc-xd)(si2) */
        pSrc[(2U * i1) + 1U] = Yc12_out;

        /* xb' = (xa+yb-xc-yd)co1 + (ya-xb-yc+xd)(si1) */
        pSrc[2U * i2] = Xb12_out;

        /* yb' = (ya-xb-yc+xd)co1 - (xa+yb-xc-yd)(si1) */
        pSrc[(2U * i2) + 1U] = Yb12_out;

        /* xd' = (xa-yb-xc+yd)co3 + (ya+xb-yc-xd)(si3) */
        pSrc[2U * i3] = Xd12_out;

        /* yd' = (ya+xb-yc-xd)co3 - (xa-yb-xc+yd)(si3) */
        pSrc[(2U * i3) + 1U] = Yd12_out;

        /*  Twiddle coefficients index modifier */
        ia1 += twidCoefModifier;

        /*  Updating input index */
        i0++;

    } while (--j);

    twidCoefModifier <<= 2U;

    /*  Calculation of second stage to excluding last stage */
    for (k = fftLen >> 2U; k > 4U; k >>= 2U)
    {
        /*  Initializations for the first stage */
        n1 = n2;
        n2 >>= 2U;
        ia1 = 0U;

        /*  Calculation of first stage */
        j = 0;
        do
        {
            /*  index calculation for the coefficients */
            ia2 = ia1 + ia1;
            ia3 = ia2 + ia1;
            co1 = pCoef[(ia1 * 2U)];
            si1 = pCoef[(ia1 * 2U) + 1U];
            co2 = pCoef[(ia2 * 2U)];
            si2 = pCoef[(ia2 * 2U) + 1U];
            co3 = pCoef[(ia3 * 2U)];
            si3 = pCoef[(ia3 * 2U) + 1U];

            /*  Twiddle coefficients index modifier */
            ia1 += twidCoefModifier;

            i0 = j;
            do
            {
                /*  index calculation for the input as, */
                /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
                i1 = i0 + n2;
                i2 = i1 + n2;
                i3 = i2 + n2;

                xaIn = pSrc[(2U * i0)];
                yaIn = pSrc[(2U * i0) + 1U];

                xbIn = pSrc[(2U * i1)];
                ybIn = pSrc[(2U * i1) + 1U];

                xcIn = pSrc[(2U * i2)];
                ycIn = pSrc[(2U * i2) + 1U];

                xdIn = pSrc[(2U * i3)];
                ydIn = pSrc[(2U * i3) + 1U];

                /* xa - xc */
                Xaminusc = xaIn - xcIn;
                /* (xb - xd) */
                Xbminusd = xbIn - xdIn;
                /* ya - yc */
                Yaminusc = yaIn - ycIn;
                /* (yb - yd) */
                Ybminusd = ybIn - ydIn;

                /* xa + xc */
                Xaplusc = xaIn + xcIn;
                /* xb + xd */
                Xbplusd = xbIn + xdIn;
                /* ya + yc */
                Yaplusc = yaIn + ycIn;
                /* yb + yd */
                Ybplusd = ybIn + ydIn;

                /* (xa - xc) + (yb - yd) */
                Xb12C_out = (Xaminusc + Ybminusd);
                /* (ya - yc) -  (xb - xd) */
                Yb12C_out = (Yaminusc - Xbminusd);
                /* xa + xc -(xb + xd) */
                Xc12C_out = (Xaplusc - Xbplusd);
                /* (ya + yc) - (yb + yd) */
                Yc12C_out = (Yaplusc - Ybplusd);
                /* (xa - xc) - (yb - yd) */
                Xd12C_out = (Xaminusc - Ybminusd);
                /* (ya - yc) +  (xb - xd) */
                Yd12C_out = (Xbminusd + Yaminusc);

                pSrc[(2U * i0)] = Xaplusc + Xbplusd;
                pSrc[(2U * i0) + 1U] = Yaplusc + Ybplusd;

                Xb12_out = Xb12C_out * co1;
                Yb12_out = Yb12C_out * co1;
                Xc12_out = Xc12C_out * co2;
                Yc12_out = Yc12C_out * co2;
                Xd12_out = Xd12C_out * co3;
                Yd12_out = Yd12C_out * co3;

                /* xb' = (xa+yb-xc-yd)co1 - (ya-xb-yc+xd)(si1) */
                // Xb12_out -= Yb12C_out * si1;
                p0 = Yb12C_out * si1;
                /* yb' = (ya-xb-yc+xd)co1 + (xa+yb-xc-yd)(si1) */
                // Yb12_out += Xb12C_out * si1;
                p1 = Xb12C_out * si1;
                /* xc' = (xa-xb+xc-xd)co2 - (ya-yb+yc-yd)(si2) */
                // Xc12_out -= Yc12C_out * si2;
                p2 = Yc12C_out * si2;
                /* yc' = (ya-yb+yc-yd)co2 + (xa-xb+xc-xd)(si2) */
                // Yc12_out += Xc12C_out * si2;
                p3 = Xc12C_out * si2;
                /* xd' = (xa-yb-xc+yd)co3 - (ya+xb-yc-xd)(si3) */
                // Xd12_out -= Yd12C_out * si3;
                p4 = Yd12C_out * si3;
                /* yd' = (ya+xb-yc-xd)co3 + (xa-yb-xc+yd)(si3) */
                // Yd12_out += Xd12C_out * si3;
                p5 = Xd12C_out * si3;

                Xb12_out += p0;
                Yb12_out -= p1;
                Xc12_out += p2;
                Yc12_out -= p3;
                Xd12_out += p4;
                Yd12_out -= p5;

                /* xc' = (xa-xb+xc-xd)co2 + (ya-yb+yc-yd)(si2) */
                pSrc[2U * i1] = Xc12_out;

                /* yc' = (ya-yb+yc-yd)co2 - (xa-xb+xc-xd)(si2) */
                pSrc[(2U * i1) + 1U] = Yc12_out;

                /* xb' = (xa+yb-xc-yd)co1 + (ya-xb-yc+xd)(si1) */
                pSrc[2U * i2] = Xb12_out;

                /* yb' = (ya-xb-yc+xd)co1 - (xa+yb-xc-yd)(si1) */
                pSrc[(2U * i2) + 1U] = Yb12_out;

                /* xd' = (xa-yb-xc+yd)co3 + (ya+xb-yc-xd)(si3) */
                pSrc[2U * i3] = Xd12_out;

                /* yd' = (ya+xb-yc-xd)co3 - (xa-yb-xc+yd)(si3) */
                pSrc[(2U * i3) + 1U] = Yd12_out;

                i0 += n1;
            } while (i0 < fftLen);
            j++;
        } while (j <= (n2 - 1U));
        twidCoefModifier <<= 2U;
    }

    j = fftLen >> 2;
    ptr1 = &pSrc[0];

    /*  Calculations of last stage */
    do
    {
        xaIn = ptr1[0];
        yaIn = ptr1[1];
        xbIn = ptr1[2];
        ybIn = ptr1[3];
        xcIn = ptr1[4];
        ycIn = ptr1[5];
        xdIn = ptr1[6];
        ydIn = ptr1[7];

        /* xa + xc */
        Xaplusc = xaIn + xcIn;

        /* xa - xc */
        Xaminusc = xaIn - xcIn;

        /* ya + yc */
        Yaplusc = yaIn + ycIn;

        /* ya - yc */
        Yaminusc = yaIn - ycIn;

        /* xb + xd */
        Xbplusd = xbIn + xdIn;

        /* yb + yd */
        Ybplusd = ybIn + ydIn;

        /* (xb-xd) */
        Xbminusd = xbIn - xdIn;

        /* (yb-yd) */
        Ybminusd = ybIn - ydIn;

        /* xa' = xa + xb + xc + xd */
        a0 = (Xaplusc + Xbplusd);
        /* ya' = ya + yb + yc + yd */
        a1 = (Yaplusc + Ybplusd);
        /* xc' = (xa-xb+xc-xd) */
        a2 = (Xaplusc - Xbplusd);
        /* yc' = (ya-yb+yc-yd) */
        a3 = (Yaplusc - Ybplusd);
        /* xb' = (xa+yb-xc-yd) */
        a4 = (Xaminusc + Ybminusd);
        /* yb' = (ya-xb-yc+xd) */
        a5 = (Yaminusc - Xbminusd);
        /* xd' = (xa-yb-xc+yd)) */
        a6 = (Xaminusc - Ybminusd);
        /* yd' = (ya+xb-yc-xd) */
        a7 = (Xbminusd + Yaminusc);

        ptr1[0] = a0;
        ptr1[1] = a1;
        ptr1[2] = a2;
        ptr1[3] = a3;
        ptr1[4] = a4;
        ptr1[5] = a5;
        ptr1[6] = a6;
        ptr1[7] = a7;

        /* increment pointer by 8 */
        ptr1 += 8U;
    } while (--j);

#else

    float32_t t1, t2, r1, r2, s1, s2;

    /* Initializations for the fft calculation */
    n2 = fftLen;
    n1 = n2;
    for (k = fftLen; k > 1U; k >>= 2U)
    {
        /*  Initializations for the fft calculation */
        n1 = n2;
        n2 >>= 2U;
        ia1 = 0U;

        /*  FFT Calculation */
        j = 0;
        do
        {
            /*  index calculation for the coefficients */
            ia2 = ia1 + ia1;
            ia3 = ia2 + ia1;
            co1 = pCoef[ia1 * 2U];
            si1 = pCoef[(ia1 * 2U) + 1U];
            co2 = pCoef[ia2 * 2U];
            si2 = pCoef[(ia2 * 2U) + 1U];
            co3 = pCoef[ia3 * 2U];
            si3 = pCoef[(ia3 * 2U) + 1U];

            /*  Twiddle coefficients index modifier */
            ia1 = ia1 + twidCoefModifier;

            i0 = j;
            do
            {
                /*  index calculation for the input as, */
                /*  pSrc[i0 + 0], pSrc[i0 + fftLen/4], pSrc[i0 + fftLen/2], pSrc[i0 + 3fftLen/4] */
                i1 = i0 + n2;
                i2 = i1 + n2;
                i3 = i2 + n2;

                /* xa + xc */
                r1 = pSrc[(2U * i0)] + pSrc[(2U * i2)];

                /* xa - xc */
                r2 = pSrc[(2U * i0)] - pSrc[(2U * i2)];

                /* ya + yc */
                s1 = pSrc[(2U * i0) + 1U] + pSrc[(2U * i2) + 1U];

                /* ya - yc */
                s2 = pSrc[(2U * i0) + 1U] - pSrc[(2U * i2) + 1U];

                /* xb + xd */
                t1 = pSrc[2U * i1] + pSrc[2U * i3];

                /* xa' = xa + xb + xc + xd */
                pSrc[2U * i0] = r1 + t1;

                /* xa + xc -(xb + xd) */
                r1 = r1 - t1;

                /* yb + yd */
                t2 = pSrc[(2U * i1) + 1U] + pSrc[(2U * i3) + 1U];

                /* ya' = ya + yb + yc + yd */
                pSrc[(2U * i0) + 1U] = s1 + t2;

                /* (ya + yc) - (yb + yd) */
                s1 = s1 - t2;

                /* (yb - yd) */
                t1 = pSrc[(2U * i1) + 1U] - pSrc[(2U * i3) + 1U];

                /* (xb - xd) */
                t2 = pSrc[2U * i1] - pSrc[2U * i3];

                /* xc' = (xa-xb+xc-xd)co2 + (ya-yb+yc-yd)(si2) */
                pSrc[2U * i1] = (r1 * co2) + (s1 * si2);

                /* yc' = (ya-yb+yc-yd)co2 - (xa-xb+xc-xd)(si2) */
                pSrc[(2U * i1) + 1U] = (s1 * co2) - (r1 * si2);

                /* (xa - xc) + (yb - yd) */
                r1 = r2 + t1;

                /* (xa - xc) - (yb - yd) */
                r2 = r2 - t1;

                /* (ya - yc) -  (xb - xd) */
                s1 = s2 - t2;

                /* (ya - yc) +  (xb - xd) */
                s2 = s2 + t2;

                /* xb' = (xa+yb-xc-yd)co1 + (ya-xb-yc+xd)(si1) */
                pSrc[2U * i2] = (r1 * co1) + (s1 * si1);

                /* yb' = (ya-xb-yc+xd)co1 - (xa+yb-xc-yd)(si1) */
                pSrc[(2U * i2) + 1U] = (s1 * co1) - (r1 * si1);

                /* xd' = (xa-yb-xc+yd)co3 + (ya+xb-yc-xd)(si3) */
                pSrc[2U * i3] = (r2 * co3) + (s2 * si3);

                /* yd' = (ya+xb-yc-xd)co3 - (xa-yb-xc+yd)(si3) */
                pSrc[(2U * i3) + 1U] = (s2 * co3) - (r2 * si3);

                i0 += n1;
            } while (i0 < fftLen);
            j++;
        } while (j <= (n2 - 1U));
        twidCoefModifier <<= 2U;
    }

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */
}

RISCV_DSP_ATTRIBUTE void riscv_split_rfft_f32(
    float32_t *pSrc,
    uint32_t fftLen,
    const float32_t *pATable,
    const float32_t *pBTable,
    float32_t *pDst,
    uint32_t modifier)
{
    uint32_t i;                                                      /* Loop Counter */
    float32_t outR, outI;                                            /* Temporary variables for output */
    const float32_t *pCoefA, *pCoefB;                                /* Temporary pointers for twiddle factors */
    float32_t CoefA1, CoefA2, CoefB1;                                /* Temporary variables for twiddle coefficients */
    float32_t *pDst1 = &pDst[2], *pDst2 = &pDst[(4U * fftLen) - 1U]; /* temp pointers for output buffer */
    float32_t *pSrc1 = &pSrc[2], *pSrc2 = &pSrc[(2U * fftLen) - 1U]; /* temp pointers for input buffer */

    /* Init coefficient pointers */
    pCoefA = &pATable[modifier * 2];
    pCoefB = &pBTable[modifier * 2];

    i = fftLen - 1U;

    while (i > 0U)
    {
        /*
          outR = (  pSrc[2 * i]             * pATable[2 * i]
                  - pSrc[2 * i + 1]         * pATable[2 * i + 1]
                  + pSrc[2 * n - 2 * i]     * pBTable[2 * i]
                  + pSrc[2 * n - 2 * i + 1] * pBTable[2 * i + 1]);

          outI = (  pIn[2 * i + 1]         * pATable[2 * i]
                  + pIn[2 * i]             * pATable[2 * i + 1]
                  + pIn[2 * n - 2 * i]     * pBTable[2 * i + 1]
                  - pIn[2 * n - 2 * i + 1] * pBTable[2 * i]);
         */

        /* read pATable[2 * i] */
        CoefA1 = *pCoefA++;
        /* pATable[2 * i + 1] */
        CoefA2 = *pCoefA;

        /* pSrc[2 * i] * pATable[2 * i] */
        outR = *pSrc1 * CoefA1;
        /* pSrc[2 * i] * CoefA2 */
        outI = *pSrc1++ * CoefA2;

        /* (pSrc[2 * i + 1] + pSrc[2 * fftLen - 2 * i + 1]) * CoefA2 */
        outR -= (*pSrc1 + *pSrc2) * CoefA2;
        /* pSrc[2 * i + 1] * CoefA1 */
        outI += *pSrc1++ * CoefA1;

        CoefB1 = *pCoefB;

        /* pSrc[2 * fftLen - 2 * i + 1] * CoefB1 */
        outI -= *pSrc2-- * CoefB1;
        /* pSrc[2 * fftLen - 2 * i] * CoefA2 */
        outI -= *pSrc2 * CoefA2;

        /* pSrc[2 * fftLen - 2 * i] * CoefB1 */
        outR += *pSrc2-- * CoefB1;

        /* write output */
        *pDst1++ = outR;
        *pDst1++ = outI;

        /* write complex conjugate output */
        *pDst2-- = -outI;
        *pDst2-- = outR;

        /* update coefficient pointer */
        pCoefB = pCoefB + (modifier * 2U);
        pCoefA = pCoefA + ((modifier * 2U) - 1U);

        i--;
    }

    pDst[2U * fftLen] = pSrc[0] - pSrc[1];
    pDst[(2U * fftLen) + 1U] = 0.0f;

    pDst[0] = pSrc[0] + pSrc[1];
    pDst[1] = 0.0f;
}

RISCV_DSP_ATTRIBUTE void riscv_rfft_f32(
    const riscv_rfft_instance_f32 *S,
    float32_t *pSrc,
    float32_t *pDst)
{
    const riscv_cfft_radix4_instance_f32 *S_CFFT = S->pCfft;

    /* Calculation of Real IFFT of input */
    if (S->ifftFlagR == 1U)
    {
        /*  Real IFFT core process */
        riscv_split_rifft_f32(pSrc, S->fftLenBy2, S->pTwiddleAReal, S->pTwiddleBReal, pDst, S->twidCoefRModifier);

        /* Complex radix-4 IFFT process */
        riscv_radix4_butterfly_inverse_f32(pDst, S_CFFT->fftLen, S_CFFT->pTwiddle, S_CFFT->twidCoefModifier, S_CFFT->onebyfftLen);

        /* Bit reversal process */
        if (S->bitReverseFlagR == 1U)
        {
            riscv_bitreversal_f32(pDst, S_CFFT->fftLen, S_CFFT->bitRevFactor, S_CFFT->pBitRevTable);
        }
    }
    else
    {
        /* Calculation of RFFT of input */

        /* Complex radix-4 FFT process */
        riscv_radix4_butterfly_f32(pSrc, S_CFFT->fftLen, S_CFFT->pTwiddle, S_CFFT->twidCoefModifier);

        /* Bit reversal process */
        if (S->bitReverseFlagR == 1U)
        {
            riscv_bitreversal_f32(pSrc, S_CFFT->fftLen, S_CFFT->bitRevFactor, S_CFFT->pBitRevTable);
        }

        /*  Real FFT core process */
        riscv_split_rfft_f32(pSrc, S->fftLenBy2, S->pTwiddleAReal, S->pTwiddleBReal, pDst, S->twidCoefRModifier);
    }
}

typedef struct
{
    uint16_t fftLen;              /**< length of the FFT. */
    const float32_t *pTwiddle;    /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable; /**< points to the bit reversal table. */
    uint16_t bitRevLength;        /**< bit reversal table length. */
} riscv_cfft_instance_f32;

typedef struct
{
    riscv_cfft_instance_f32 Sint;  /**< Internal CFFT structure. */
    uint16_t fftLenRFFT;           /**< length of the real sequence */
    const float32_t *pTwiddleRFFT; /**< Twiddle factors real stage  */
} riscv_rfft_fast_instance_f32;

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len16 = {
    16, twiddleCoef_16, riscvBitRevIndexTable16, RISCVBITREVINDEXTABLE_16_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len32 = {
    32, twiddleCoef_32, riscvBitRevIndexTable32, RISCVBITREVINDEXTABLE_32_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len64 = {
    64, twiddleCoef_64, riscvBitRevIndexTable64, RISCVBITREVINDEXTABLE_64_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len128 = {
    128, twiddleCoef_128, riscvBitRevIndexTable128, RISCVBITREVINDEXTABLE_128_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len256 = {
    256, twiddleCoef_256, riscvBitRevIndexTable256, RISCVBITREVINDEXTABLE_256_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len512 = {
    512, twiddleCoef_512, riscvBitRevIndexTable512, RISCVBITREVINDEXTABLE_512_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len1024 = {
    1024, twiddleCoef_1024, riscvBitRevIndexTable1024, RISCVBITREVINDEXTABLE_1024_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len2048 = {
    2048, twiddleCoef_2048, riscvBitRevIndexTable2048, RISCVBITREVINDEXTABLE_2048_TABLE_LENGTH};

const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len4096 = {
    4096, twiddleCoef_4096, riscvBitRevIndexTable4096, RISCVBITREVINDEXTABLE_4096_TABLE_LENGTH};

#define FFTINIT(EXT, SIZE)                                          \
    S->bitRevLength = riscv_cfft_sR_##EXT##_len##SIZE.bitRevLength; \
    S->pBitRevTable = riscv_cfft_sR_##EXT##_len##SIZE.pBitRevTable; \
    S->pTwiddle = riscv_cfft_sR_##EXT##_len##SIZE.pTwiddle;

#define CFFTINIT_F32(LEN, LENTWIDDLE)                                                        \
    RISCV_DSP_ATTRIBUTE riscv_status riscv_cfft_init_##LEN##_f32(riscv_cfft_instance_f32 *S) \
    {                                                                                        \
        /*  Initialise the default riscv status */                                           \
        riscv_status status = RISCV_MATH_SUCCESS;                                            \
                                                                                             \
        /*  Initialise the FFT length */                                                     \
        S->fftLen = LEN;                                                                     \
                                                                                             \
        /*  Initialise the Twiddle coefficient pointer */                                    \
        S->pTwiddle = NULL;                                                                  \
                                                                                             \
        FFTINIT(f32, LEN);                                                                   \
                                                                                             \
        return (status);                                                                     \
    }

/**
  @brief         Initialization function for the cfft f32 function with 4096 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(4096, 4096)

/**
  @brief         Initialization function for the cfft f32 function with 2048 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(2048, 1024)

/**
  @brief         Initialization function for the cfft f32 function with 1024 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(1024, 1024)

/**
  @brief         Initialization function for the cfft f32 function with 512 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(512, 256)

/**
  @brief         Initialization function for the cfft f32 function with 256 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(256, 256)

/**
  @brief         Initialization function for the cfft f32 function with 128 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(128, 64)

/**
  @brief         Initialization function for the cfft f32 function with 64 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(64, 64)

/**
  @brief         Initialization function for the cfft f32 function with 32 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(32, 16)

/**
  @brief         Initialization function for the cfft f32 function with 16 samples
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h
 */
CFFTINIT_F32(16, 16)

/**
  @brief         Generic initialization function for the cfft f32 function
  @param[in,out] S              points to an instance of the floating-point CFFT structure
  @param[in]     fftLen         fft length (number of complex samples)
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected

  @par          Use of this function is mandatory only for the MVE version of the FFT.
                Other versions can still initialize directly the data structure using
                variables declared in riscv_const_structs.h

  @par
                This function should be used only if you don't know the FFT sizes that
                you'll need at build time. The use of this function will prevent the
                linker from removing the FFT tables that are not needed and the library
                code size will be bigger than needed.

  @par
                If you use NMSIS-DSP as a static library, and if you know the FFT sizes
                that you need at build time, then it is better to use the initialization
                functions defined for each FFT size.
 */
riscv_status riscv_cfft_init_f32(
    riscv_cfft_instance_f32 *S,
    uint16_t fftLen)
{

    /*  Initialise the default riscv status */
    riscv_status status = RISCV_MATH_SUCCESS;

    /*  Initializations of Instance structure depending on the FFT length */
    switch (fftLen)
    {
        /*  Initializations of structure parameters for 4096 point FFT */
    case 4096U:
        /*  Initialise the bit reversal table modifier */
        status = riscv_cfft_init_4096_f32(S);
        break;

        /*  Initializations of structure parameters for 2048 point FFT */
    case 2048U:
        /*  Initialise the bit reversal table modifier */
        status = riscv_cfft_init_2048_f32(S);
        break;

        /*  Initializations of structure parameters for 1024 point FFT */
    case 1024U:
        /*  Initialise the bit reversal table modifier */
        status = riscv_cfft_init_1024_f32(S);
        break;

        /*  Initializations of structure parameters for 512 point FFT */
    case 512U:
        /*  Initialise the bit reversal table modifier */
        status = riscv_cfft_init_512_f32(S);
        break;

    case 256U:
        status = riscv_cfft_init_256_f32(S);
        break;

    case 128U:
        status = riscv_cfft_init_128_f32(S);
        break;

    case 64U:
        status = riscv_cfft_init_64_f32(S);
        break;

    case 32U:
        status = riscv_cfft_init_32_f32(S);
        break;

    case 16U:
        /*  Initializations of structure parameters for 16 point FFT */
        status = riscv_cfft_init_16_f32(S);
        break;

    default:
        /*  Reporting argument error if fftSize is not valid value */
        status = RISCV_MATH_ARGUMENT_ERROR;
        break;
    }

    return (status);
}

riscv_status riscv_cfft_init_4096_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_2048_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_1024_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_512_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_256_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_128_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_64_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_32_f32(riscv_cfft_instance_f32 *S);
riscv_status riscv_cfft_init_16_f32(riscv_cfft_instance_f32 *S);

/**
  @ingroup RealFFT
 */

/**
  @addtogroup RealFFTF32
  @{
 */

/**
  @brief         Initialization function for the 32pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_32_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_16_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }

    S->fftLenRFFT = 32U;
    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_32;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 64pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_64_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_32_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 64U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_64;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 128pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_128_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_64_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 128;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_128;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 256pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
*/

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_256_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_128_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 256U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_256;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 512pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_512_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_256_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 512U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_512;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 1024pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_1024_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_512_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 1024U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_1024;

    return RISCV_MATH_SUCCESS;
}

/**
  @brief         Initialization function for the 2048pt floating-point real FFT.
  @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */
RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_2048_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_1024_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 2048U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_2048;

    return RISCV_MATH_SUCCESS;
}

/**
* @brief         Initialization function for the 4096pt floating-point real FFT.
* @param[in,out] S  points to an riscv_rfft_fast_instance_f32 structure
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : Operation successful
                   - \ref RISCV_MATH_ARGUMENT_ERROR : an error is detected
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_4096_f32(riscv_rfft_fast_instance_f32 *S)
{

    riscv_status status;

    if (!S)
        return RISCV_MATH_ARGUMENT_ERROR;

    status = riscv_cfft_init_2048_f32(&(S->Sint));
    if (status != RISCV_MATH_SUCCESS)
    {
        return (status);
    }
    S->fftLenRFFT = 4096U;

    S->pTwiddleRFFT = (float32_t *)twiddleCoef_rfft_4096;

    return RISCV_MATH_SUCCESS;
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_rfft_fast_init_f32(
    riscv_rfft_fast_instance_f32 *S,
    uint16_t fftLen)
{
    riscv_status status;

    switch (fftLen)
    {
    case 4096U:
        status = riscv_rfft_fast_init_4096_f32(S);
        break;
    case 2048U:
        status = riscv_rfft_fast_init_2048_f32(S);
        break;
    case 1024U:
        status = riscv_rfft_fast_init_1024_f32(S);
        break;
    case 512U:
        status = riscv_rfft_fast_init_512_f32(S);
        break;
    case 256U:
        status = riscv_rfft_fast_init_256_f32(S);
        break;
    case 128U:
        status = riscv_rfft_fast_init_128_f32(S);
        break;
    case 64U:
        status = riscv_rfft_fast_init_64_f32(S);
        break;
    case 32U:
        status = riscv_rfft_fast_init_32_f32(S);
        break;
    default:
        return (RISCV_MATH_ARGUMENT_ERROR);
        break;
    }

    return (status);
}

static void merge_rfft_f32(
    const riscv_rfft_fast_instance_f32 *S,
    const float32_t *p,
    float32_t *pOut)
{
    int32_t k;                                 /* Loop Counter */
    float32_t twR, twI;                        /* RFFT Twiddle coefficients */
    const float32_t *pCoeff = S->pTwiddleRFFT; /* Points to RFFT Twiddle factors */
    const float32_t *pA = p;                   /* increasing pointer */
    const float32_t *pB = p;                   /* decreasing pointer */
    float32_t xAR, xAI, xBR, xBI;              /* temporary variables */
    float32_t t1a, t1b, r, s, t, u;            /* temporary variables */

    k = (S->Sint).fftLen - 1;

    xAR = pA[0];
    xAI = pA[1];

    pCoeff += 2;

    *pOut++ = 0.5f * (xAR + xAI);
    *pOut++ = 0.5f * (xAR - xAI);

    pB = p + 2 * k;
    pA += 2;

    while (k > 0)
    {
        /* G is half of the frequency complex spectrum */
        // for k = 2:N
        //     Xk(k) = 1/2 * (G(k) + conj(G(N-k+2)) + Tw(k)*( G(k) - conj(G(N-k+2))));
        xBI = pB[1];
        xBR = pB[0];
        xAR = pA[0];
        xAI = pA[1];

        twR = *pCoeff++;
        twI = *pCoeff++;

        t1a = xAR - xBR;
        t1b = xAI + xBI;

        r = twR * t1a;
        s = twI * t1b;
        t = twI * t1a;
        u = twR * t1b;

        // real(tw * (xA - xB)) = twR * (xAR - xBR) - twI * (xAI - xBI);
        // imag(tw * (xA - xB)) = twI * (xAR - xBR) + twR * (xAI - xBI);
        *pOut++ = 0.5f * (xAR + xBR - r - s); // xAR
        *pOut++ = 0.5f * (xAI - xBI + t - u); // xAI

        pA += 2;
        pB -= 2;
        k--;
    }
}

RISCV_DSP_ATTRIBUTE void riscv_radix8_butterfly_f32(
    float32_t *pSrc,
    uint16_t fftLen,
    const float32_t *pCoef,
    uint16_t twidCoefModifier)
{
    uint32_t ia1, ia2, ia3, ia4, ia5, ia6, ia7;
    uint32_t i1, i2, i3, i4, i5, i6, i7, i8;
    uint32_t id;
    uint32_t n1, n2, j;

    float32_t r1, r2, r3, r4, r5, r6, r7, r8;
    float32_t t1, t2;
    float32_t s1, s2, s3, s4, s5, s6, s7, s8;
    float32_t p1, p2, p3, p4;
    float32_t co2, co3, co4, co5, co6, co7, co8;
    float32_t si2, si3, si4, si5, si6, si7, si8;
    const float32_t C81 = 0.70710678118f;

    n2 = fftLen;

    do
    {
        n1 = n2;
        n2 = n2 >> 3;
        i1 = 0;

        do
        {
            i2 = i1 + n2;
            i3 = i2 + n2;
            i4 = i3 + n2;
            i5 = i4 + n2;
            i6 = i5 + n2;
            i7 = i6 + n2;
            i8 = i7 + n2;
            r1 = pSrc[2 * i1] + pSrc[2 * i5];
            r5 = pSrc[2 * i1] - pSrc[2 * i5];
            r2 = pSrc[2 * i2] + pSrc[2 * i6];
            r6 = pSrc[2 * i2] - pSrc[2 * i6];
            r3 = pSrc[2 * i3] + pSrc[2 * i7];
            r7 = pSrc[2 * i3] - pSrc[2 * i7];
            r4 = pSrc[2 * i4] + pSrc[2 * i8];
            r8 = pSrc[2 * i4] - pSrc[2 * i8];
            t1 = r1 - r3;
            r1 = r1 + r3;
            r3 = r2 - r4;
            r2 = r2 + r4;
            pSrc[2 * i1] = r1 + r2;
            pSrc[2 * i5] = r1 - r2;
            r1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
            s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
            r2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
            s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
            s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
            s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
            r4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
            s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
            t2 = r1 - s3;
            r1 = r1 + s3;
            s3 = r2 - r4;
            r2 = r2 + r4;
            pSrc[2 * i1 + 1] = r1 + r2;
            pSrc[2 * i5 + 1] = r1 - r2;
            pSrc[2 * i3] = t1 + s3;
            pSrc[2 * i7] = t1 - s3;
            pSrc[2 * i3 + 1] = t2 - r3;
            pSrc[2 * i7 + 1] = t2 + r3;
            r1 = (r6 - r8) * C81;
            r6 = (r6 + r8) * C81;
            r2 = (s6 - s8) * C81;
            s6 = (s6 + s8) * C81;
            t1 = r5 - r1;
            r5 = r5 + r1;
            r8 = r7 - r6;
            r7 = r7 + r6;
            t2 = s5 - r2;
            s5 = s5 + r2;
            s8 = s7 - s6;
            s7 = s7 + s6;
            pSrc[2 * i2] = r5 + s7;
            pSrc[2 * i8] = r5 - s7;
            pSrc[2 * i6] = t1 + s8;
            pSrc[2 * i4] = t1 - s8;
            pSrc[2 * i2 + 1] = s5 - r7;
            pSrc[2 * i8 + 1] = s5 + r7;
            pSrc[2 * i6 + 1] = t2 - r8;
            pSrc[2 * i4 + 1] = t2 + r8;

            i1 += n1;
        } while (i1 < fftLen);

        if (n2 < 8)
            break;

        ia1 = 0;
        j = 1;

        do
        {
            /*  index calculation for the coefficients */
            id = ia1 + twidCoefModifier;
            ia1 = id;
            ia2 = ia1 + id;
            ia3 = ia2 + id;
            ia4 = ia3 + id;
            ia5 = ia4 + id;
            ia6 = ia5 + id;
            ia7 = ia6 + id;

            co2 = pCoef[2 * ia1];
            co3 = pCoef[2 * ia2];
            co4 = pCoef[2 * ia3];
            co5 = pCoef[2 * ia4];
            co6 = pCoef[2 * ia5];
            co7 = pCoef[2 * ia6];
            co8 = pCoef[2 * ia7];
            si2 = pCoef[2 * ia1 + 1];
            si3 = pCoef[2 * ia2 + 1];
            si4 = pCoef[2 * ia3 + 1];
            si5 = pCoef[2 * ia4 + 1];
            si6 = pCoef[2 * ia5 + 1];
            si7 = pCoef[2 * ia6 + 1];
            si8 = pCoef[2 * ia7 + 1];

            i1 = j;

            do
            {
                /*  index calculation for the input */
                i2 = i1 + n2;
                i3 = i2 + n2;
                i4 = i3 + n2;
                i5 = i4 + n2;
                i6 = i5 + n2;
                i7 = i6 + n2;
                i8 = i7 + n2;
                r1 = pSrc[2 * i1] + pSrc[2 * i5];
                r5 = pSrc[2 * i1] - pSrc[2 * i5];
                r2 = pSrc[2 * i2] + pSrc[2 * i6];
                r6 = pSrc[2 * i2] - pSrc[2 * i6];
                r3 = pSrc[2 * i3] + pSrc[2 * i7];
                r7 = pSrc[2 * i3] - pSrc[2 * i7];
                r4 = pSrc[2 * i4] + pSrc[2 * i8];
                r8 = pSrc[2 * i4] - pSrc[2 * i8];
                t1 = r1 - r3;
                r1 = r1 + r3;
                r3 = r2 - r4;
                r2 = r2 + r4;
                pSrc[2 * i1] = r1 + r2;
                r2 = r1 - r2;
                s1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
                s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
                s2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
                s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
                s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
                s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
                s4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
                s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
                t2 = s1 - s3;
                s1 = s1 + s3;
                s3 = s2 - s4;
                s2 = s2 + s4;
                r1 = t1 + s3;
                t1 = t1 - s3;
                pSrc[2 * i1 + 1] = s1 + s2;
                s2 = s1 - s2;
                s1 = t2 - r3;
                t2 = t2 + r3;
                p1 = co5 * r2;
                p2 = si5 * s2;
                p3 = co5 * s2;
                p4 = si5 * r2;
                pSrc[2 * i5] = p1 + p2;
                pSrc[2 * i5 + 1] = p3 - p4;
                p1 = co3 * r1;
                p2 = si3 * s1;
                p3 = co3 * s1;
                p4 = si3 * r1;
                pSrc[2 * i3] = p1 + p2;
                pSrc[2 * i3 + 1] = p3 - p4;
                p1 = co7 * t1;
                p2 = si7 * t2;
                p3 = co7 * t2;
                p4 = si7 * t1;
                pSrc[2 * i7] = p1 + p2;
                pSrc[2 * i7 + 1] = p3 - p4;
                r1 = (r6 - r8) * C81;
                r6 = (r6 + r8) * C81;
                s1 = (s6 - s8) * C81;
                s6 = (s6 + s8) * C81;
                t1 = r5 - r1;
                r5 = r5 + r1;
                r8 = r7 - r6;
                r7 = r7 + r6;
                t2 = s5 - s1;
                s5 = s5 + s1;
                s8 = s7 - s6;
                s7 = s7 + s6;
                r1 = r5 + s7;
                r5 = r5 - s7;
                r6 = t1 + s8;
                t1 = t1 - s8;
                s1 = s5 - r7;
                s5 = s5 + r7;
                s6 = t2 - r8;
                t2 = t2 + r8;
                p1 = co2 * r1;
                p2 = si2 * s1;
                p3 = co2 * s1;
                p4 = si2 * r1;
                pSrc[2 * i2] = p1 + p2;
                pSrc[2 * i2 + 1] = p3 - p4;
                p1 = co8 * r5;
                p2 = si8 * s5;
                p3 = co8 * s5;
                p4 = si8 * r5;
                pSrc[2 * i8] = p1 + p2;
                pSrc[2 * i8 + 1] = p3 - p4;
                p1 = co6 * r6;
                p2 = si6 * s6;
                p3 = co6 * s6;
                p4 = si6 * r6;
                pSrc[2 * i6] = p1 + p2;
                pSrc[2 * i6 + 1] = p3 - p4;
                p1 = co4 * t1;
                p2 = si4 * t2;
                p3 = co4 * t2;
                p4 = si4 * t1;
                pSrc[2 * i4] = p1 + p2;
                pSrc[2 * i4 + 1] = p3 - p4;

                i1 += n1;
            } while (i1 < fftLen);

            j++;
        } while (j < n2);

        twidCoefModifier <<= 3;
    } while (n2 > 7);
}

static void riscv_cfft_radix8by2_f32(riscv_cfft_instance_f32 *S, float32_t *p1)
{
    uint32_t L = S->fftLen;
    float32_t *pCol1, *pCol2, *pMid1, *pMid2;
    float32_t *p2 = p1 + L;
    const float32_t *tw = (float32_t *)S->pTwiddle;
    float32_t t1[4], t2[4], t3[4], t4[4], twR, twI;
    float32_t m0, m1, m2, m3;
    uint32_t l;

    pCol1 = p1;
    pCol2 = p2;

    /* Define new length */
    L >>= 1;

    /* Initialize mid pointers */
    pMid1 = p1 + L;
    pMid2 = p2 + L;

    /* do two dot Fourier transform */
    for (l = L >> 2; l > 0; l--)
    {
        t1[0] = p1[0];
        t1[1] = p1[1];
        t1[2] = p1[2];
        t1[3] = p1[3];

        t2[0] = p2[0];
        t2[1] = p2[1];
        t2[2] = p2[2];
        t2[3] = p2[3];

        t3[0] = pMid1[0];
        t3[1] = pMid1[1];
        t3[2] = pMid1[2];
        t3[3] = pMid1[3];

        t4[0] = pMid2[0];
        t4[1] = pMid2[1];
        t4[2] = pMid2[2];
        t4[3] = pMid2[3];

        *p1++ = t1[0] + t2[0];
        *p1++ = t1[1] + t2[1];
        *p1++ = t1[2] + t2[2];
        *p1++ = t1[3] + t2[3]; /* col 1 */

        t2[0] = t1[0] - t2[0];
        t2[1] = t1[1] - t2[1];
        t2[2] = t1[2] - t2[2];
        t2[3] = t1[3] - t2[3]; /* for col 2 */

        *pMid1++ = t3[0] + t4[0];
        *pMid1++ = t3[1] + t4[1];
        *pMid1++ = t3[2] + t4[2];
        *pMid1++ = t3[3] + t4[3]; /* col 1 */

        t4[0] = t4[0] - t3[0];
        t4[1] = t4[1] - t3[1];
        t4[2] = t4[2] - t3[2];
        t4[3] = t4[3] - t3[3]; /* for col 2 */

        twR = *tw++;
        twI = *tw++;

        /* multiply by twiddle factors */
        m0 = t2[0] * twR;
        m1 = t2[1] * twI;
        m2 = t2[1] * twR;
        m3 = t2[0] * twI;

        /* R  =  R  *  Tr - I * Ti */
        *p2++ = m0 + m1;
        /* I  =  I  *  Tr + R * Ti */
        *p2++ = m2 - m3;

        /* use vertical symmetry */
        /*  0.9988 - 0.0491i <==> -0.0491 - 0.9988i */
        m0 = t4[0] * twI;
        m1 = t4[1] * twR;
        m2 = t4[1] * twI;
        m3 = t4[0] * twR;

        *pMid2++ = m0 - m1;
        *pMid2++ = m2 + m3;

        twR = *tw++;
        twI = *tw++;

        m0 = t2[2] * twR;
        m1 = t2[3] * twI;
        m2 = t2[3] * twR;
        m3 = t2[2] * twI;

        *p2++ = m0 + m1;
        *p2++ = m2 - m3;

        m0 = t4[2] * twI;
        m1 = t4[3] * twR;
        m2 = t4[3] * twI;
        m3 = t4[2] * twR;

        *pMid2++ = m0 - m1;
        *pMid2++ = m2 + m3;
    }

    /* first col */
    riscv_radix8_butterfly_f32(pCol1, L, (float32_t *)S->pTwiddle, 2U);

    /* second col */
    riscv_radix8_butterfly_f32(pCol2, L, (float32_t *)S->pTwiddle, 2U);
}

static void riscv_cfft_radix8by4_f32(riscv_cfft_instance_f32 *S, float32_t *p1)
{
    uint32_t L = S->fftLen >> 1;
    float32_t *pCol1, *pCol2, *pCol3, *pCol4, *pEnd1, *pEnd2, *pEnd3, *pEnd4;
    const float32_t *tw2, *tw3, *tw4;
    float32_t *p2 = p1 + L;
    float32_t *p3 = p2 + L;
    float32_t *p4 = p3 + L;
    float32_t t2[4], t3[4], t4[4], twR, twI;
    float32_t p1ap3_0, p1sp3_0, p1ap3_1, p1sp3_1;
    float32_t m0, m1, m2, m3;
    uint32_t l, twMod2, twMod3, twMod4;

    pCol1 = p1; /* points to real values by default */
    pCol2 = p2;
    pCol3 = p3;
    pCol4 = p4;
    pEnd1 = p2 - 1; /* points to imaginary values by default */
    pEnd2 = p3 - 1;
    pEnd3 = p4 - 1;
    pEnd4 = pEnd3 + L;

    tw2 = tw3 = tw4 = (float32_t *)S->pTwiddle;

    L >>= 1;

    /* do four dot Fourier transform */

    twMod2 = 2;
    twMod3 = 4;
    twMod4 = 6;

    /* TOP */
    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];

    /* col 2 */
    t2[0] = p1sp3_0 + p2[1] - p4[1];
    t2[1] = p1sp3_1 - p2[0] + p4[0];
    /* col 3 */
    t3[0] = p1ap3_0 - p2[0] - p4[0];
    t3[1] = p1ap3_1 - p2[1] - p4[1];
    /* col 4 */
    t4[0] = p1sp3_0 - p2[1] + p4[1];
    t4[1] = p1sp3_1 + p2[0] - p4[0];
    /* col 1 */
    *p1++ = p1ap3_0 + p2[0] + p4[0];
    *p1++ = p1ap3_1 + p2[1] + p4[1];

    /* Twiddle factors are ones */
    *p2++ = t2[0];
    *p2++ = t2[1];
    *p3++ = t3[0];
    *p3++ = t3[1];
    *p4++ = t4[0];
    *p4++ = t4[1];

    tw2 += twMod2;
    tw3 += twMod3;
    tw4 += twMod4;

    for (l = (L - 2) >> 1; l > 0; l--)
    {
        /* TOP */
        p1ap3_0 = p1[0] + p3[0];
        p1sp3_0 = p1[0] - p3[0];
        p1ap3_1 = p1[1] + p3[1];
        p1sp3_1 = p1[1] - p3[1];
        /* col 2 */
        t2[0] = p1sp3_0 + p2[1] - p4[1];
        t2[1] = p1sp3_1 - p2[0] + p4[0];
        /* col 3 */
        t3[0] = p1ap3_0 - p2[0] - p4[0];
        t3[1] = p1ap3_1 - p2[1] - p4[1];
        /* col 4 */
        t4[0] = p1sp3_0 - p2[1] + p4[1];
        t4[1] = p1sp3_1 + p2[0] - p4[0];
        /* col 1 - top */
        *p1++ = p1ap3_0 + p2[0] + p4[0];
        *p1++ = p1ap3_1 + p2[1] + p4[1];

        /* BOTTOM */
        p1ap3_1 = pEnd1[-1] + pEnd3[-1];
        p1sp3_1 = pEnd1[-1] - pEnd3[-1];
        p1ap3_0 = pEnd1[0] + pEnd3[0];
        p1sp3_0 = pEnd1[0] - pEnd3[0];
        /* col 2 */
        t2[2] = pEnd2[0] - pEnd4[0] + p1sp3_1;
        t2[3] = pEnd1[0] - pEnd3[0] - pEnd2[-1] + pEnd4[-1];
        /* col 3 */
        t3[2] = p1ap3_1 - pEnd2[-1] - pEnd4[-1];
        t3[3] = p1ap3_0 - pEnd2[0] - pEnd4[0];
        /* col 4 */
        t4[2] = pEnd2[0] - pEnd4[0] - p1sp3_1;
        t4[3] = pEnd4[-1] - pEnd2[-1] - p1sp3_0;
        /* col 1 - Bottom */
        *pEnd1-- = p1ap3_0 + pEnd2[0] + pEnd4[0];
        *pEnd1-- = p1ap3_1 + pEnd2[-1] + pEnd4[-1];

        /* COL 2 */
        /* read twiddle factors */
        twR = *tw2++;
        twI = *tw2++;
        /* multiply by twiddle factors */
        /*  let    Z1 = a + i(b),   Z2 = c + i(d) */
        /*   =>  Z1 * Z2  =  (a*c - b*d) + i(b*c + a*d) */

        /* Top */
        m0 = t2[0] * twR;
        m1 = t2[1] * twI;
        m2 = t2[1] * twR;
        m3 = t2[0] * twI;

        *p2++ = m0 + m1;
        *p2++ = m2 - m3;
        /* use vertical symmetry col 2 */
        /* 0.9997 - 0.0245i  <==>  0.0245 - 0.9997i */
        /* Bottom */
        m0 = t2[3] * twI;
        m1 = t2[2] * twR;
        m2 = t2[2] * twI;
        m3 = t2[3] * twR;

        *pEnd2-- = m0 - m1;
        *pEnd2-- = m2 + m3;

        /* COL 3 */
        twR = tw3[0];
        twI = tw3[1];
        tw3 += twMod3;
        /* Top */
        m0 = t3[0] * twR;
        m1 = t3[1] * twI;
        m2 = t3[1] * twR;
        m3 = t3[0] * twI;

        *p3++ = m0 + m1;
        *p3++ = m2 - m3;
        /* use vertical symmetry col 3 */
        /* 0.9988 - 0.0491i  <==>  -0.9988 - 0.0491i */
        /* Bottom */
        m0 = -t3[3] * twR;
        m1 = t3[2] * twI;
        m2 = t3[2] * twR;
        m3 = t3[3] * twI;

        *pEnd3-- = m0 - m1;
        *pEnd3-- = m3 - m2;

        /* COL 4 */
        twR = tw4[0];
        twI = tw4[1];
        tw4 += twMod4;
        /* Top */
        m0 = t4[0] * twR;
        m1 = t4[1] * twI;
        m2 = t4[1] * twR;
        m3 = t4[0] * twI;

        *p4++ = m0 + m1;
        *p4++ = m2 - m3;
        /* use vertical symmetry col 4 */
        /* 0.9973 - 0.0736i  <==>  -0.0736 + 0.9973i */
        /* Bottom */
        m0 = t4[3] * twI;
        m1 = t4[2] * twR;
        m2 = t4[2] * twI;
        m3 = t4[3] * twR;

        *pEnd4-- = m0 - m1;
        *pEnd4-- = m2 + m3;
    }

    /* MIDDLE */
    /* Twiddle factors are */
    /*  1.0000  0.7071-0.7071i  -1.0000i  -0.7071-0.7071i */
    p1ap3_0 = p1[0] + p3[0];
    p1sp3_0 = p1[0] - p3[0];
    p1ap3_1 = p1[1] + p3[1];
    p1sp3_1 = p1[1] - p3[1];

    /* col 2 */
    t2[0] = p1sp3_0 + p2[1] - p4[1];
    t2[1] = p1sp3_1 - p2[0] + p4[0];
    /* col 3 */
    t3[0] = p1ap3_0 - p2[0] - p4[0];
    t3[1] = p1ap3_1 - p2[1] - p4[1];
    /* col 4 */
    t4[0] = p1sp3_0 - p2[1] + p4[1];
    t4[1] = p1sp3_1 + p2[0] - p4[0];
    /* col 1 - Top */
    *p1++ = p1ap3_0 + p2[0] + p4[0];
    *p1++ = p1ap3_1 + p2[1] + p4[1];

    /* COL 2 */
    twR = tw2[0];
    twI = tw2[1];

    m0 = t2[0] * twR;
    m1 = t2[1] * twI;
    m2 = t2[1] * twR;
    m3 = t2[0] * twI;

    *p2++ = m0 + m1;
    *p2++ = m2 - m3;
    /* COL 3 */
    twR = tw3[0];
    twI = tw3[1];

    m0 = t3[0] * twR;
    m1 = t3[1] * twI;
    m2 = t3[1] * twR;
    m3 = t3[0] * twI;

    *p3++ = m0 + m1;
    *p3++ = m2 - m3;
    /* COL 4 */
    twR = tw4[0];
    twI = tw4[1];

    m0 = t4[0] * twR;
    m1 = t4[1] * twI;
    m2 = t4[1] * twR;
    m3 = t4[0] * twI;

    *p4++ = m0 + m1;
    *p4++ = m2 - m3;

    /* first col */
    riscv_radix8_butterfly_f32(pCol1, L, (float32_t *)S->pTwiddle, 4U);

    /* second col */
    riscv_radix8_butterfly_f32(pCol2, L, (float32_t *)S->pTwiddle, 4U);

    /* third col */
    riscv_radix8_butterfly_f32(pCol3, L, (float32_t *)S->pTwiddle, 4U);

    /* fourth col */
    riscv_radix8_butterfly_f32(pCol4, L, (float32_t *)S->pTwiddle, 4U);
}

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_32(
    uint32_t *pSrc,
    const uint16_t bitRevLen,
    const uint16_t *pBitRevTab)
{
    uint32_t a, b, i, tmp;

    for (i = 0; i < bitRevLen;)
    {
        a = pBitRevTab[i] >> 2;
        b = pBitRevTab[i + 1] >> 2;

        // real
        tmp = pSrc[a];
        pSrc[a] = pSrc[b];
        pSrc[b] = tmp;

        // complex
        tmp = pSrc[a + 1];
        pSrc[a + 1] = pSrc[b + 1];
        pSrc[b + 1] = tmp;

        i += 2;
    }
}

RISCV_DSP_ATTRIBUTE void riscv_cfft_f32(
    const riscv_cfft_instance_f32 *S,
    float32_t *p1,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag)
{
    uint32_t L = S->fftLen, l;
    float32_t invL, *pSrc;

    if (ifftFlag == 1U)
    {
        /* Conjugate input data */
        pSrc = p1 + 1;
        for (l = 0; l < L; l++)
        {
            *pSrc = -*pSrc;
            pSrc += 2;
        }
    }

    switch (L)
    {
    case 16:
    case 128:
    case 1024:
        riscv_cfft_radix8by2_f32((riscv_cfft_instance_f32 *)S, p1);
        break;
    case 32:
    case 256:
    case 2048:
        riscv_cfft_radix8by4_f32((riscv_cfft_instance_f32 *)S, p1);
        break;
    case 64:
    case 512:
    case 4096:
        riscv_radix8_butterfly_f32(p1, L, (float32_t *)S->pTwiddle, 1);
        break;
    }

    if (bitReverseFlag)
        riscv_bitreversal_32((uint32_t *)p1, S->bitRevLength, S->pBitRevTable);

    if (ifftFlag == 1U)
    {
        invL = 1.0f / (float32_t)L;

        /* Conjugate and scale output data */
        pSrc = p1;
        for (l = 0; l < L; l++)
        {
            *pSrc++ *= invL;
            *pSrc = -(*pSrc) * invL;
            pSrc++;
        }
    }
}

static void stage_rfft_f32(
    const riscv_rfft_fast_instance_f32 *S,
    const float32_t *p,
    float32_t *pOut)
{
    int32_t k;                                 /* Loop Counter */
    float32_t twR, twI;                        /* RFFT Twiddle coefficients */
    const float32_t *pCoeff = S->pTwiddleRFFT; /* Points to RFFT Twiddle factors */
    const float32_t *pA = p;                   /* increasing pointer */
    const float32_t *pB = p;                   /* decreasing pointer */
    float32_t xAR, xAI, xBR, xBI;              /* temporary variables */
    float32_t t1a, t1b;                        /* temporary variables */
    float32_t p0, p1, p2, p3;                  /* temporary variables */

    k = (S->Sint).fftLen - 1;

    /* Pack first and last sample of the frequency domain together */

    xBR = pB[0];
    xBI = pB[1];
    xAR = pA[0];
    xAI = pA[1];

    twR = *pCoeff++;
    twI = *pCoeff++;

    // U1 = XA(1) + XB(1); % It is real
    t1a = xBR + xAR;

    // U2 = XB(1) - XA(1); % It is imaginary
    t1b = xBI + xAI;

    // real(tw * (xB - xA)) = twR * (xBR - xAR) - twI * (xBI - xAI);
    // imag(tw * (xB - xA)) = twI * (xBR - xAR) + twR * (xBI - xAI);
    *pOut++ = 0.5f * (t1a + t1b);
    *pOut++ = 0.5f * (t1a - t1b);

    // XA(1) = 1/2*( U1 - imag(U2) +  i*( U1 +imag(U2) ));
    pB = p + 2 * k;
    pA += 2;

    do
    {
        /*
           function X = my_split_rfft(X, ifftFlag)
           % X is a series of real numbers
           L  = length(X);
           XC = X(1:2:end) +i*X(2:2:end);
           XA = fft(XC);
           XB = conj(XA([1 end:-1:2]));
           TW = i*exp(-2*pi*i*[0:L/2-1]/L).';
           for l = 2:L/2
              XA(l) = 1/2 * (XA(l) + XB(l) + TW(l) * (XB(l) - XA(l)));
           end
           XA(1) = 1/2* (XA(1) + XB(1) + TW(1) * (XB(1) - XA(1))) + i*( 1/2*( XA(1) + XB(1) + i*( XA(1) - XB(1))));
           X = XA;
        */

        xBI = pB[1];
        xBR = pB[0];
        xAR = pA[0];
        xAI = pA[1];

        twR = *pCoeff++;
        twI = *pCoeff++;

        t1a = xBR - xAR;
        t1b = xBI + xAI;

        // real(tw * (xB - xA)) = twR * (xBR - xAR) - twI * (xBI - xAI);
        // imag(tw * (xB - xA)) = twI * (xBR - xAR) + twR * (xBI - xAI);
        p0 = twR * t1a;
        p1 = twI * t1a;
        p2 = twR * t1b;
        p3 = twI * t1b;

        *pOut++ = 0.5f * (xAR + xBR + p0 + p3); // xAR
        *pOut++ = 0.5f * (xAI - xBI + p1 - p2); // xAI

        pA += 2;
        pB -= 2;
        k--;
    } while (k > 0);
}

RISCV_DSP_ATTRIBUTE void riscv_rfft_fast_f32(
    const riscv_rfft_fast_instance_f32 *S,
    float32_t *p,
    float32_t *pOut,
    uint8_t ifftFlag)
{
    const riscv_cfft_instance_f32 *Sint = &(S->Sint);

    /* Calculation of Real FFT */
    if (ifftFlag)
    {
        /*  Real FFT compression */
        merge_rfft_f32(S, p, pOut);
        /* Complex radix-4 IFFT process */
        riscv_cfft_f32(Sint, pOut, ifftFlag, 1);
    }
    else
    {
        /* Calculation of RFFT of input */
        riscv_cfft_f32(Sint, p, ifftFlag, 1);

        /*  Real FFT extraction */
        stage_rfft_f32(S, p, pOut);
    }
}