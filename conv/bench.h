#pragma once

#include "real_coef.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

uint64_t wtime(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) == -1)
        return 0;

    return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

#ifndef READ_CYCLE
/** Read run cycle of cpu */
#define READ_CYCLE wtime
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

#define ARRAYA_SIZE_F32 1024
#define ARRAYB_SIZE_F32 1024

static uint32_t firstIndex = 4;
static uint32_t numPoints = 128;
static float32_t test_conv_input_f32_A[ARRAYA_SIZE_F32] = {};
static float32_t test_conv_input_f32_B[ARRAYB_SIZE_F32] = {};

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_f32(
    const float32_t *pSrcA,
    uint32_t srcALen,
    const float32_t *pSrcB,
    uint32_t srcBLen,
    float32_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{
#if defined(RISCV_MATH_DSP) || defined(RISCV_MATH_VECTOR)
    const float32_t *pIn1 = pSrcA;  /* InputA pointer */
    const float32_t *pIn2 = pSrcB;  /* InputB pointer */
    float32_t *pOut = pDst;         /* Output pointer */
    const float32_t *px;            /* Intermediate inputA pointer */
    const float32_t *py;            /* Intermediate inputB pointer */
    const float32_t *pSrc1, *pSrc2; /* Intermediate pointers */
    float32_t sum;                  /* Accumulator */
    uint32_t j, k, count, blkCnt, check;
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    riscv_status status;                        /* Status of Partial convolution */

#if defined(RISCV_MATH_LOOPUNROLL)
    float32_t acc0, acc1, acc2, acc3; /* Accumulator */
    float32_t x0, x1, x2, x3, c0;     /* Temporary variables */
#endif

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = ((int32_t)check - blockSize3) - (blockSize1 + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc1 = pIn2 + firstIndex;
        py = pSrc1;

        /* ------------------------
         * Stage1 process
         * ----------------------*/

        /* The first stage starts here */
        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0.0f;

#if defined(RISCV_MATH_VECTOR)
            size_t vblkCnt = count; /* Loop counter */
            size_t l;
            vfloat32m8_t vx, vy;
            vfloat32m8_t vsum;
            ptrdiff_t bstride = -4;
            l = __riscv_vsetvlmax_e32m8();
            vsum = __riscv_vfmv_v_f_f32m8(0.0f, l);
            for (; (l = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_f32m8(px, l);
                px += l;
                vy = __riscv_vlse32_v_f32m8(py, bstride, l);
                py -= l;
                vsum = __riscv_vfmacc_vv_f32m8(vsum, vx, vy, l);
            }
            l = __riscv_vsetvl_e32m8(1);
            vfloat32m1_t temp00m1 = __riscv_vfmv_v_f_f32m1(0.0f, l);
            l = __riscv_vsetvlmax_e32m8();
            temp00m1 = __riscv_vfredusum_vs_f32m8_f32m1(vsum, temp00m1, l);
            sum += __riscv_vfmv_f_s_f32m1_f32(temp00m1);
#else
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
                /* x[0] * y[srcBLen - 1] */
                sum += *px++ * *py--;

                /* x[1] * y[srcBLen - 2] */
                sum += *px++ * *py--;

                /* x[2] * y[srcBLen - 3] */
                sum += *px++ * *py--;

                /* x[3] * y[srcBLen - 4] */
                sum += *px++ * *py--;

                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;

#else

            /* Initialize k with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */
            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                sum += *px++ * *py--;

                /* Decrement loop counter */
                k--;
            }
#endif /*defined (RISCV_MATH_VECTOR)*/
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = sum;

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc1;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR)
        blkCnt = blockSize2;

        while (blkCnt > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0.0f;

            /* srcBLen number of MACS should be performed */
            size_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vfloat32m8_t vx, vy;
            vfloat32m8_t vsum;
            ptrdiff_t bstride = -4;

            l = __riscv_vsetvlmax_e32m8();
            vsum = __riscv_vfmv_v_f_f32m8(0.0f, l);
            for (; (l = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_f32m8(px, l);
                px += l;
                vy = __riscv_vlse32_v_f32m8(py, bstride, l);
                py -= l;
                vsum = __riscv_vfmacc_vv_f32m8(vsum, vx, vy, l);
            }
            l = __riscv_vsetvl_e32m8(1);
            vfloat32m1_t temp00m1 = __riscv_vfmv_v_f_f32m1(0.0f, l);
            l = __riscv_vsetvlmax_e32m8();
            temp00m1 = __riscv_vfredusum_vs_f32m8_f32m1(vsum, temp00m1, l);
            sum += __riscv_vfmv_f_s_f32m1_f32(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = sum;

            /* Increment the MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = pIn1 + count;
            py = pSrc2;

            /* Decrement the loop counter */
            blkCnt--;
        }
#else
        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            blkCnt = ((uint32_t)blockSize2 >> 2U);

            while (blkCnt > 0U)
            {
                /* Set all accumulators to zero */
                acc0 = 0.0f;
                acc1 = 0.0f;
                acc2 = 0.0f;
                acc3 = 0.0f;

                /* read x[0], x[1], x[2] samples */
                x0 = *px++;
                x1 = *px++;
                x2 = *px++;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 3 samples. */
                do
                {
                    /* Read y[srcBLen - 1] sample */
                    c0 = *py--;
                    /* Read x[3] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[0] * y[srcBLen - 1] */
                    acc0 += x0 * c0;
                    /* acc1 +=  x[1] * y[srcBLen - 1] */
                    acc1 += x1 * c0;
                    /* acc2 +=  x[2] * y[srcBLen - 1] */
                    acc2 += x2 * c0;
                    /* acc3 +=  x[3] * y[srcBLen - 1] */
                    acc3 += x3 * c0;

                    /* Read y[srcBLen - 2] sample */
                    c0 = *py--;
                    /* Read x[4] sample */
                    x0 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[1] * y[srcBLen - 2] */
                    acc0 += x1 * c0;
                    /* acc1 +=  x[2] * y[srcBLen - 2] */
                    acc1 += x2 * c0;
                    /* acc2 +=  x[3] * y[srcBLen - 2] */
                    acc2 += x3 * c0;
                    /* acc3 +=  x[4] * y[srcBLen - 2] */
                    acc3 += x0 * c0;

                    /* Read y[srcBLen - 3] sample */
                    c0 = *py--;
                    /* Read x[5] sample */
                    x1 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[2] * y[srcBLen - 3] */
                    acc0 += x2 * c0;
                    /* acc1 +=  x[3] * y[srcBLen - 2] */
                    acc1 += x3 * c0;
                    /* acc2 +=  x[4] * y[srcBLen - 2] */
                    acc2 += x0 * c0;
                    /* acc3 +=  x[5] * y[srcBLen - 2] */
                    acc3 += x1 * c0;

                    /* Read y[srcBLen - 4] sample */
                    c0 = *py--;
                    /* Read x[6] sample */
                    x2 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[3] * y[srcBLen - 4] */
                    acc0 += x3 * c0;
                    /* acc1 +=  x[4] * y[srcBLen - 4] */
                    acc1 += x0 * c0;
                    /* acc2 +=  x[5] * y[srcBLen - 4] */
                    acc2 += x1 * c0;
                    /* acc3 +=  x[6] * y[srcBLen - 4] */
                    acc3 += x2 * c0;

                } while (--k);

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;

                while (k > 0U)
                {
                    /* Read y[srcBLen - 5] sample */
                    c0 = *py--;
                    /* Read x[7] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[4] * y[srcBLen - 5] */
                    acc0 += x0 * c0;
                    /* acc1 +=  x[5] * y[srcBLen - 5] */
                    acc1 += x1 * c0;
                    /* acc2 +=  x[6] * y[srcBLen - 5] */
                    acc2 += x2 * c0;
                    /* acc3 +=  x[7] * y[srcBLen - 5] */
                    acc3 += x3 * c0;

                    /* Reuse the present samples for the next MAC */
                    x0 = x1;
                    x1 = x2;
                    x2 = x3;

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = acc0;
                *pOut++ = acc1;
                *pOut++ = acc2;
                *pOut++ = acc3;

                /* Increment the pointer pIn1 index, count by 4 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }

            /* Loop unrolling: Compute remaining outputs */
            blkCnt = (uint32_t)blockSize2 % 0x4U;

#else

            /* Initialize blkCnt with number of samples */
            blkCnt = blockSize2;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0.0f;

#if defined(RISCV_MATH_LOOPUNROLL)

                /* Loop unrolling: Compute 4 outputs at a time */
                k = srcBLen >> 2U;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulates */
                    sum += *px++ * *py--;
                    sum += *px++ * *py--;
                    sum += *px++ * *py--;
                    sum += *px++ * *py--;

                    /* Decrement loop counter */
                    k--;
                }

                /* Loop unrolling: Compute remaining outputs */
                k = srcBLen % 0x4U;

#else

                /* Initialize blkCnt with number of samples */
                k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += *px++ * *py--;

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = sum;

                /* Increment MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0.0f;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += *px++ * *py--;

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = sum;

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }
#endif /*defined (RISCV_MATH_VECTOR)*/

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The blockSize3 variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* -------------------
         * Stage3 process
         * ------------------*/

        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0.0f;
#if defined(RISCV_MATH_VECTOR)
            size_t vblkCnt = blockSize3; /* Loop counter */
            size_t l;
            vfloat32m8_t vx, vy;
            vfloat32m8_t vsum;
            ptrdiff_t bstride = -4;
            l = __riscv_vsetvlmax_e32m8();
            vsum = __riscv_vfmv_v_f_f32m8(0.0f, l);
            for (; (l = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_f32m8(px, l);
                px += l;
                vy = __riscv_vlse32_v_f32m8(py, bstride, l);
                py -= l;
                vsum = __riscv_vfmacc_vv_f32m8(vsum, vx, vy, l);
            }
            l = __riscv_vsetvl_e32m8(1);
            vfloat32m1_t temp00m1 = __riscv_vfmv_v_f_f32m1(0.0f, l);
            l = __riscv_vsetvlmax_e32m8();
            temp00m1 = __riscv_vfredusum_vs_f32m8_f32m1(vsum, temp00m1, l);
            sum += __riscv_vfmv_f_s_f32m1_f32(temp00m1);
            sum += __riscv_vfmv_f_s_f32m1_f32(temp00m1);
#else

#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
                /* sum += x[srcALen - srcBLen + 1] * y[srcBLen - 1] */
                sum += *px++ * *py--;

                /* sum += x[srcALen - srcBLen + 2] * y[srcBLen - 2] */
                sum += *px++ * *py--;

                /* sum += x[srcALen - srcBLen + 3] * y[srcBLen - 3] */
                sum += *px++ * *py--;

                /* sum += x[srcALen - srcBLen + 4] * y[srcBLen - 4] */
                sum += *px++ * *py--;

                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;

#else

            /* Initialize blkCnt with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum += *px++ * *py--;

                /* Decrement loop counter */
                k--;
            }
#endif /*defined (RISCV_MATH_VECTOR)*/
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = sum;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#else
    /* alternate version for CM0_FAMILY */

    const float32_t *pIn1 = pSrcA; /* InputA pointer */
    const float32_t *pIn2 = pSrcB; /* InputB pointer */
    float32_t sum;                 /* Accumulator */
    uint32_t i, j;                 /* Loop counters */
    riscv_status status;           /* Status of Partial convolution */
    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* Loop to calculate convolution for output length number of values */
        for (i = firstIndex; i <= (firstIndex + numPoints - 1); i++)
        {
            /* Initialize sum with zero to carry on MAC operations */
            sum = 0.0f;

            /* Loop to perform MAC operations according to convolution equation */
            for (j = 0U; j <= i; j++)
            {
                /* Check the array limitations */
                if (((i - j) < srcBLen) && (j < srcALen))
                {
                    /* z[i] += x[i-j] * y[j] */
                    sum += (pIn1[j] * pIn2[i - j]);
                }
            }

            /* Store the output in the destination buffer */
            pDst[i] = sum;
        }

        /* Set status as RISCV_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#endif /* defined(RISCV_MATH_DSP) */
}

#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)

typedef int8_t q7_t;
typedef int16_t q15_t;
typedef int32_t q31_t;
typedef int64_t q63_t;

#define Q31_MAX ((q31_t)(0x7FFFFFFFL))
#define Q15_MAX ((q15_t)(0x7FFF))
#define Q7_MAX ((q7_t)(0x7F))
#define Q31_MIN ((q31_t)(0x80000000L))
#define Q15_MIN ((q15_t)(0x8000))
#define Q7_MIN ((q7_t)(0x80))

#define ARRAYA_SIZE_Q7 1024
#define ARRAYB_SIZE_Q7 1024

static q7_t test_conv_input_q7_A[ARRAYA_SIZE_Q7] = {};
static q7_t test_conv_input_q7_B[ARRAYB_SIZE_Q7] = {};

static void generate_rand_q7(q7_t *src, int length)
{
    do_srand();
    for (int i = 0; i < length; i++)
    {
        src[i] = (q7_t)(rand() % Q7_MAX - Q7_MAX / 2);
    }
}

static void generate_rand_q15(q15_t *src, int length)
{
    do_srand();
    for (int i = 0; i < length; i++)
    {
        src[i] = (q15_t)(rand() % Q15_MAX - Q15_MAX / 2);
    }
}

static void generate_rand_q31(q31_t *src, int length)
{
    do_srand();
    for (int i = 0; i < length; i++)
    {
        src[i] = (q31_t)(rand() % Q31_MAX - Q31_MAX / 2);
    }
}

__STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat)
{
    if ((sat >= 1U) && (sat <= 32U))
    {
        const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
        const int32_t min = -1 - max;
        if (val > max)
        {
            return max;
        }
        else if (val < min)
        {
            return min;
        }
    }
    return val;
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_q7(
    const q7_t *pSrcA,
    uint32_t srcALen,
    const q7_t *pSrcB,
    uint32_t srcBLen,
    q7_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{

#if defined(RISCV_MATH_DSP) || defined(RISCV_MATH_VECTOR)

    const q7_t *pIn1;                           /* InputA pointer */
    const q7_t *pIn2;                           /* InputB pointer */
    q7_t *pOut = pDst;                          /* Output pointer */
    const q7_t *px;                             /* Intermediate inputA pointer */
    const q7_t *py;                             /* Intermediate inputB pointer */
    const q7_t *pSrc1, *pSrc2;                  /* Intermediate pointers */
    q31_t sum;                                  /* Accumulator */
    uint32_t j, k, count, blkCnt, check;        /* Loop counters */
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    riscv_status status;                        /* Status of Partial convolution */

#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t acc0, acc1, acc2, acc3; /* Accumulator */
    q31_t input1, input2;         /* Temporary input variables */
    q15_t in1, in2;               /* Temporary input variables */
    q7_t x0, x1, x2, x3, c0, c1;  /* Temporary variables to hold state and coefficient values */
#endif

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = (int32_t)check - ((blockSize3 + blockSize1) + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + firstIndex;
        py = pSrc2;

        /* ------------------------
         * Stage1 process
         * ----------------------*/
        /* The first stage starts here */
#if defined(RISCV_MATH_VECTOR)
        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint8m4_t vx, vy;
            vint16m1_t temp00m1;
            ptrdiff_t bstride = -1;
            l = __riscv_vsetvl_e16m1(vblkCnt);
            temp00m1 = __riscv_vmv_v_x_i16m1(0, l);
            for (; (l = __riscv_vsetvl_e8m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle8_v_i8m4(px, l);
                px += l;
                vy = __riscv_vlse8_v_i8m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i16m8_i16m1(__riscv_vwmul_vv_i16m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i16m1_i16(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }
#else
        /* The first stage starts here */
        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
                /* x[0] , x[1] */
                in1 = (q15_t)*px++;
                in2 = (q15_t)*px++;
                input1 = __RV_PKBB16(in2, in1);

                /* y[srcBLen - 1] , y[srcBLen - 2] */
                in1 = (q15_t)*py--;
                in2 = (q15_t)*py--;
                input2 = __RV_PKBB16(in2, in1);

                /* x[0] * y[srcBLen - 1] */
                /* x[1] * y[srcBLen - 2] */
                sum = __SMLAD(input1, input2, sum);

                /* x[2] , x[3] */
                in1 = (q15_t)*px++;
                in2 = (q15_t)*px++;
                input1 = __RV_PKBB16(in2, in1);

                /* y[srcBLen - 3] , y[srcBLen - 4] */
                in1 = (q15_t)*py--;
                in2 = (q15_t)*py--;
                input2 = __RV_PKBB16(in2, in1);

                /* x[2] * y[srcBLen - 3] */
                /* x[3] * y[srcBLen - 4] */
                sum = __SMLAD(input1, input2, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;

#else

            /* Initialize k with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                sum += ((q31_t)*px++ * *py--);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }
#endif /*defined (RISCV_MATH_VECTOR)*/
        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is the index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR)
        blkCnt = blockSize2;

        while (blkCnt > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            uint32_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vint8m4_t vx, vy;
            vint16m1_t temp00m1;
            ptrdiff_t bstride = -1;
            l = __riscv_vsetvl_e16m1(1);
            temp00m1 = __riscv_vmv_v_x_i16m1(0, l);
            for (; (l = __riscv_vsetvl_e8m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle8_v_i8m4(px, l);
                px += l;
                vy = __riscv_vlse8_v_i8m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i16m8_i16m1(__riscv_vwmul_vv_i16m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i16m1_i16(temp00m1);

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(sum >> 7U, 8));

            /* Increment the MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = pIn1 + count;
            py = pSrc2;

            /* Decrement loop counter */
            blkCnt--;
        }
#else
        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            blkCnt = ((uint32_t)blockSize2 >> 2U);

            while (blkCnt > 0U)
            {
                /* Set all accumulators to zero */
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;
                acc3 = 0;

                /* read x[0], x[1], x[2] samples */
                x0 = *px++;
                x1 = *px++;
                x2 = *px++;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 3 samples. */
                do
                {
                    /* Read y[srcBLen - 1] sample */
                    c0 = *py--;
                    /* Read y[srcBLen - 2] sample */
                    c1 = *py--;

                    /* Read x[3] sample */
                    x3 = *px++;

                    /* x[0] and x[1] are packed */
                    in1 = (q15_t)x0;
                    in2 = (q15_t)x1;

                    input1 = __RV_PKBB16(in2, in1);

                    /* y[srcBLen - 1]   and y[srcBLen - 2] are packed */
                    in1 = (q15_t)c0;
                    in2 = (q15_t)c1;

                    input2 = __RV_PKBB16(in2, in1);

                    /* acc0 += x[0] * y[srcBLen - 1] + x[1] * y[srcBLen - 2]  */
                    acc0 = __SMLAD(input1, input2, acc0);

                    /* x[1] and x[2] are packed */
                    in1 = (q15_t)x1;
                    in2 = (q15_t)x2;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc1 += x[1] * y[srcBLen - 1] + x[2] * y[srcBLen - 2]  */
                    acc1 = __SMLAD(input1, input2, acc1);

                    /* x[2] and x[3] are packed */
                    in1 = (q15_t)x2;
                    in2 = (q15_t)x3;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc2 += x[2] * y[srcBLen - 1] + x[3] * y[srcBLen - 2]  */
                    acc2 = __SMLAD(input1, input2, acc2);

                    /* Read x[4] sample */
                    x0 = *px++;

                    /* x[3] and x[4] are packed */
                    in1 = (q15_t)x3;
                    in2 = (q15_t)x0;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc3 += x[3] * y[srcBLen - 1] + x[4] * y[srcBLen - 2]  */
                    acc3 = __SMLAD(input1, input2, acc3);

                    /* Read y[srcBLen - 3] sample */
                    c0 = *py--;
                    /* Read y[srcBLen - 4] sample */
                    c1 = *py--;

                    /* Read x[5] sample */
                    x1 = *px++;

                    /* x[2] and x[3] are packed */
                    in1 = (q15_t)x2;
                    in2 = (q15_t)x3;

                    input1 = __RV_PKBB16(in2, in1);

                    /* y[srcBLen - 3] and y[srcBLen - 4] are packed */
                    in1 = (q15_t)c0;
                    in2 = (q15_t)c1;

                    input2 = __RV_PKBB16(in2, in1);

                    /* acc0 += x[2] * y[srcBLen - 3] + x[3] * y[srcBLen - 4]  */
                    acc0 = __SMLAD(input1, input2, acc0);

                    /* x[3] and x[4] are packed */
                    in1 = (q15_t)x3;
                    in2 = (q15_t)x0;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc1 += x[3] * y[srcBLen - 3] + x[4] * y[srcBLen - 4]  */
                    acc1 = __SMLAD(input1, input2, acc1);

                    /* x[4] and x[5] are packed */
                    in1 = (q15_t)x0;
                    in2 = (q15_t)x1;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc2 += x[4] * y[srcBLen - 3] + x[5] * y[srcBLen - 4]  */
                    acc2 = __SMLAD(input1, input2, acc2);

                    /* Read x[6] sample */
                    x2 = *px++;

                    /* x[5] and x[6] are packed */
                    in1 = (q15_t)x1;
                    in2 = (q15_t)x2;

                    input1 = __RV_PKBB16(in2, in1);

                    /* acc3 += x[5] * y[srcBLen - 3] + x[6] * y[srcBLen - 4]  */
                    acc3 = __SMLAD(input1, input2, acc3);

                } while (--k);

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;

                while (k > 0U)
                {
                    /* Read y[srcBLen - 5] sample */
                    c0 = *py--;
                    /* Read x[7] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[4] * y[srcBLen - 5] */
                    acc0 += ((q31_t)x0 * c0);
                    /* acc1 +=  x[5] * y[srcBLen - 5] */
                    acc1 += ((q31_t)x1 * c0);
                    /* acc2 +=  x[6] * y[srcBLen - 5] */
                    acc2 += ((q31_t)x2 * c0);
                    /* acc3 +=  x[7] * y[srcBLen - 5] */
                    acc3 += ((q31_t)x3 * c0);

                    /* Reuse the present samples for the next MAC */
                    x0 = x1;
                    x1 = x2;
                    x2 = x3;

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q7_t)(__SSAT(acc0 >> 7, 8));
                *pOut++ = (q7_t)(__SSAT(acc1 >> 7, 8));
                *pOut++ = (q7_t)(__SSAT(acc2 >> 7, 8));
                *pOut++ = (q7_t)(__SSAT(acc3 >> 7, 8));

                /* Increment the pointer pIn1 index, count by 4 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }

            /* Loop unrolling: Compute remaining outputs */
            blkCnt = (uint32_t)blockSize2 & 0x3U;

#else

            /* Initialize blkCnt with number of samples */
            blkCnt = blockSize2;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

#if defined(RISCV_MATH_LOOPUNROLL)

                /* Loop unrolling: Compute 4 outputs at a time */
                k = srcBLen >> 2U;

                while (k > 0U)
                {
                    /* Reading two inputs of SrcA buffer and packing */
                    in1 = (q15_t)*px++;
                    in2 = (q15_t)*px++;
                    input1 = __RV_PKBB16(in2, in1);

                    /* Reading two inputs of SrcB buffer and packing */
                    in1 = (q15_t)*py--;
                    in2 = (q15_t)*py--;
                    input2 = __RV_PKBB16(in2, in1);

                    /* Perform the multiply-accumulate */
                    sum = __SMLAD(input1, input2, sum);

                    /* Reading two inputs of SrcA buffer and packing */
                    in1 = (q15_t)*px++;
                    in2 = (q15_t)*px++;
                    input1 = __RV_PKBB16(in2, in1);

                    /* Reading two inputs of SrcB buffer and packing */
                    in1 = (q15_t)*py--;
                    in2 = (q15_t)*py--;
                    input2 = __RV_PKBB16(in2, in1);

                    /* Perform the multiply-accumulate */
                    sum = __SMLAD(input1, input2, sum);

                    /* Decrement loop counter */
                    k--;
                }

                /* Loop unrolling: Compute remaining outputs */
                k = srcBLen & 0x3U;

#else

                /* Initialize blkCnt with number of samples */
                k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += ((q31_t)*px++ * *py--);

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

                /* Increment the pointer pIn1 index, count by 1 */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += ((q31_t)*px++ * *py--);

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }
#endif /*defined (RISCV_MATH_VECTOR)*/

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The count variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* -------------------
         * Stage3 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR)

        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint8m4_t vx, vy;
            vint16m1_t temp00m1;
            ptrdiff_t bstride = -1;
            l = __riscv_vsetvl_e16m1(1);
            temp00m1 = __riscv_vmv_v_x_i16m1(0, l);
            for (; (l = __riscv_vsetvl_e8m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle8_v_i8m4(px, l);
                px += l;
                vy = __riscv_vlse8_v_i8m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i16m8_i16m1(__riscv_vwmul_vv_i16m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i16m1_i16(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }
#else
        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
                /* Reading two inputs, x[srcALen - srcBLen + 1] and x[srcALen - srcBLen + 2] of SrcA buffer and packing */
                in1 = (q15_t)*px++;
                in2 = (q15_t)*px++;
                input1 = __RV_PKBB16(in2, in1);

                /* Reading two inputs, y[srcBLen - 1] and y[srcBLen - 2] of SrcB buffer and packing */
                in1 = (q15_t)*py--;
                in2 = (q15_t)*py--;
                input2 = __RV_PKBB16(in2, in1);

                /* sum += x[srcALen - srcBLen + 1] * y[srcBLen - 1] */
                /* sum += x[srcALen - srcBLen + 2] * y[srcBLen - 2] */
                sum = __SMLAD(input1, input2, sum);

                /* Reading two inputs, x[srcALen - srcBLen + 3] and x[srcALen - srcBLen + 4] of SrcA buffer and packing */
                in1 = (q15_t)*px++;
                in2 = (q15_t)*px++;
                input1 = __RV_PKBB16(in2, in1);

                /* Reading two inputs, y[srcBLen - 3] and y[srcBLen - 4] of SrcB buffer and packing */
                in1 = (q15_t)*py--;
                in2 = (q15_t)*py--;
                input2 = __RV_PKBB16(in2, in1);

                /* sum += x[srcALen - srcBLen + 3] * y[srcBLen - 3] */
                /* sum += x[srcALen - srcBLen + 4] * y[srcBLen - 4] */
                sum = __SMLAD(input1, input2, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;

#else

            /* Initialize blkCnt with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum += ((q31_t)*px++ * *py--);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(sum >> 7, 8));

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#else
    /* alternate version for CM0_FAMILY */

    const q7_t *pIn1 = pSrcA; /* InputA pointer */
    const q7_t *pIn2 = pSrcB; /* InputB pointer */
    q31_t sum;                /* Accumulator */
    uint32_t i, j;            /* Loop counters */
    riscv_status status;      /* Status of Partial convolution */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* Loop to calculate convolution for output length number of values */
        for (i = firstIndex; i <= (firstIndex + numPoints - 1); i++)
        {
            /* Initialize sum with zero to carry on MAC operations */
            sum = 0;

            /* Loop to perform MAC operations according to convolution equation */
            for (j = 0U; j <= i; j++)
            {
                /* Check the array limitations */
                if (((i - j) < srcBLen) && (j < srcALen))
                {
                    /* z[i] += x[i-j] * y[j] */
                    sum += ((q15_t)pIn1[j] * (pIn2[i - j]));
                }
            }

            /* Store the output in the destination buffer */
            pDst[i] = (q7_t)__SSAT((sum >> 7U), 8U);
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#endif /* defined(RISCV_MATH_DSP) || defined (RISCV_MATH_VECTOR) */
}

#define ARRAYA_SIZE_Q15 1024
#define ARRAYB_SIZE_Q15 1024

static q15_t test_conv_input_q15_A[ARRAYA_SIZE_Q15] = {};
static q15_t test_conv_input_q15_B[ARRAYB_SIZE_Q15] = {};

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_q15(
    const q15_t *pSrcA,
    uint32_t srcALen,
    const q15_t *pSrcB,
    uint32_t srcBLen,
    q15_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{

#if defined(RISCV_MATH_DSP) || defined(RISCV_MATH_VECTOR)

    const q15_t *pIn1;                          /* InputA pointer */
    const q15_t *pIn2;                          /* InputB pointer */
    q15_t *pOut = pDst;                         /* Output pointer */
    q63_t sum, acc0, acc1, acc2, acc3;          /* Accumulator */
    const q15_t *px;                            /* Intermediate inputA pointer */
    const q15_t *py;                            /* Intermediate inputB pointer */
    const q15_t *pSrc1, *pSrc2;                 /* Intermediate pointers */
    q31_t x0, x1, x2, x3, c0;                   /* Temporary input variables to hold state and coefficient values */
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    uint32_t j, k, count, blkCnt, check;
    riscv_status status; /* Status of Partial convolution */
    q15_t tmp0, tmp1, tmp2, tmp3;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
    q63_t px64, py64;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = (int32_t)check - ((blockSize3 + blockSize1) + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + firstIndex;
        py = pSrc2;

        /* ------------------------
         * Stage1 process
         * ----------------------*/

        /* For loop unrolling by 4, this stage is divided into two. */
        /* First part of this stage computes the MAC operations less than 4 */
        /* Second part of this stage computes the MAC operations greater than or equal to 4 */
#if defined(RISCV_MATH_VECTOR)
        while (blockSize1 > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint16m4_t vx, vy;
            vint32m1_t temp00m1;
            ptrdiff_t bstride = -2;
            l = __riscv_vsetvl_e32m1(vblkCnt);
            temp00m1 = __riscv_vmv_v_x_i32m1(0, l);
            for (; (l = __riscv_vsetvl_e16m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle16_v_i16m4(px, l);
                px += l;
                vy = __riscv_vlse16_v_i16m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i32m8_i32m1(__riscv_vwmul_vv_i32m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i32m1_i32(temp00m1);

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = pIn2 + count;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }
#else
        /* The first part of the stage starts here */
        while ((count < 4U) && (blockSize1 > 0))
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Loop over number of MAC operations between
             * inputA samples and inputB samples */
            k = count;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                sum = __SMLALD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* The second part of the stage starts here */
        /* The internal loop, over count, is unrolled by 4 */
        /* To, read the last two inputB samples using SIMD:
         * y[srcBLen] and y[srcBLen-1] coefficients, py is decremented by 1 */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 32)
        py = py - 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 32) */

        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count >> 2U;

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
               a second loop below computes MACs for the remaining 1 to 3 samples. */
            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                px64 = read_q15x4_ia((q15_t **)&px);
                tmp0 = *py--;
                tmp1 = *py--;
                tmp2 = *py--;
                tmp3 = *py--;
                py64 = __RV_PKBB32(__RV_PKBB16(tmp3, tmp2), __RV_PKBB16(tmp1, tmp0));
                sum = __SMLALD(px64, py64, sum);
#else
                /* x[0], x[1] are multiplied with y[srcBLen - 1], y[srcBLen - 2] respectively */
                sum = __SMLALDX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);
                /* x[2], x[3] are multiplied with y[srcBLen - 3], y[srcBLen - 4] respectively */
                sum = __SMLALDX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }

            /* For the next MAC operations, the pointer py is used without SIMD
             * So, py is incremented by 1 */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 32)
            py = py + 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 32) */

            /* If the count is not a multiple of 4, compute any remaining MACs here.
               No loop unrolling is used. */
            k = count & 0x3U;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                sum = __SMLALD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            py = ++pSrc2;
#else
            py = ++pSrc2 - 1U;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is the index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR)
        blkCnt = blockSize2;

        while (blkCnt > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            uint32_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vint16m4_t vx, vy;
            vint32m1_t temp00m1;
            ptrdiff_t bstride = -2;
            l = __riscv_vsetvl_e32m1(vblkCnt);
            temp00m1 = __riscv_vmv_v_x_i32m1(0, l);
            for (; (l = __riscv_vsetvl_e16m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle16_v_i16m4(px, l);
                px += l;
                vy = __riscv_vlse16_v_i16m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i32m8_i32m1(__riscv_vwmul_vv_i32m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i32m1_i32(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT(sum >> 15, 16));

            /* Increment the MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = pIn1 + count;
            py = pSrc2;

            /* Decrement the loop counter */
            blkCnt--;
        }
#else
        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
            /* Loop unrolling: Compute 4 outputs at a time */
            blkCnt = ((uint32_t)blockSize2 >> 2U);

            while (blkCnt > 0U)
            {
                py = py - 1U;

                /* Set all accumulators to zero */
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;
                acc3 = 0;

                /* read x[0], x[1] samples */
                x0 = read_q15x2((q15_t *)px);
                /* read x[1], x[2] samples */
                x1 = __RV_PKBT16(*(px + 2), x0);
                px += 2U;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 3 samples. */
                do
                {
                    /* Read the last two inputB samples using SIMD:
                     * y[srcBLen - 1] and y[srcBLen - 2] */
                    c0 = read_q15x2_da((q15_t **)&py);

                    /* acc0 +=  x[0] * y[srcBLen - 1] + x[1] * y[srcBLen - 2] */
                    acc0 = __SMLALDX(x0, c0, acc0);

                    /* acc1 +=  x[1] * y[srcBLen - 1] + x[2] * y[srcBLen - 2] */
                    acc1 = __SMLALDX(x1, c0, acc1);

                    /* Read x[2], x[3] */
                    x2 = read_q15x2((q15_t *)px);

                    /* Read x[3], x[4] */
                    x3 = __RV_PKBT16(*(px + 2), x2);

                    /* acc2 +=  x[2] * y[srcBLen - 1] + x[3] * y[srcBLen - 2] */
                    acc2 = __SMLALDX(x2, c0, acc2);

                    /* acc3 +=  x[3] * y[srcBLen - 1] + x[4] * y[srcBLen - 2] */
                    acc3 = __SMLALDX(x3, c0, acc3);

                    /* Read y[srcBLen - 3] and y[srcBLen - 4] */
                    c0 = read_q15x2_da((q15_t **)&py);

                    /* acc0 +=  x[2] * y[srcBLen - 3] + x[3] * y[srcBLen - 4] */
                    acc0 = __SMLALDX(x2, c0, acc0);

                    /* acc1 +=  x[3] * y[srcBLen - 3] + x[4] * y[srcBLen - 4] */
                    acc1 = __SMLALDX(x3, c0, acc1);

                    /* Read x[4], x[5] */
                    x0 = read_q15x2((q15_t *)px + 2);

                    /* Read x[5], x[6] */
                    x1 = __RV_PKBT16(*(px + 4), x0);
                    px += 4U;

                    /* acc2 +=  x[4] * y[srcBLen - 3] + x[5] * y[srcBLen - 4] */
                    acc2 = __SMLALDX(x0, c0, acc2);

                    /* acc3 +=  x[5] * y[srcBLen - 3] + x[6] * y[srcBLen - 4] */
                    acc3 = __SMLALDX(x1, c0, acc3);

                } while (--k);

                /* For the next MAC operations, SIMD is not used
                 * So, the 16 bit pointer if inputB, py is updated */

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;

                if (k == 1U)
                {
                    /* Read y[srcBLen - 5] */
                    c0 = *(py + 1);
                    c0 = c0 & 0x0000FFFF;

                    /* Read x[7] */
                    x3 = read_q15x2((q15_t *)px);
                    px++;

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLALD(x0, c0, acc0);
                    acc1 = __SMLALD(x1, c0, acc1);
                    acc2 = __SMLALDX(x1, c0, acc2);
                    acc3 = __SMLALDX(x3, c0, acc3);
                }

                if (k == 2U)
                {
                    /* Read y[srcBLen - 5], y[srcBLen - 6] */
                    c0 = read_q15x2((q15_t *)py);

                    /* Read x[7], x[8] */
                    x3 = read_q15x2((q15_t *)px);

                    /* Read x[9] */
                    x2 = __RV_PKBT16(*(px + 2), x3);
                    px += 2U;

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLALDX(x0, c0, acc0);
                    acc1 = __SMLALDX(x1, c0, acc1);
                    acc2 = __SMLALDX(x3, c0, acc2);
                    acc3 = __SMLALDX(x2, c0, acc3);
                }

                if (k == 3U)
                {
                    /* Read y[srcBLen - 5], y[srcBLen - 6] */
                    c0 = read_q15x2((q15_t *)py);

                    /* Read x[7], x[8] */
                    x3 = read_q15x2((q15_t *)px);

                    /* Read x[9] */
                    x2 = __RV_PKBT16(*(px + 2), x3);

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLALDX(x0, c0, acc0);
                    acc1 = __SMLALDX(x1, c0, acc1);
                    acc2 = __SMLALDX(x3, c0, acc2);
                    acc3 = __SMLALDX(x2, c0, acc3);

                    c0 = *(py - 1);
                    c0 = c0 & 0x0000FFFF;

                    /* Read x[10] */
                    x3 = read_q15x2((q15_t *)px + 2);
                    px += 3U;

                    /* Perform the multiply-accumulates */
                    acc0 = __SMLALDX(x1, c0, acc0);
                    acc1 = __SMLALD(x2, c0, acc1);
                    acc2 = __SMLALDX(x2, c0, acc2);
                    acc3 = __SMLALDX(x3, c0, acc3);
                }

                /* Store the results in the accumulators in the destination buffer. */
                {
                    int32_t sat0 = __SSAT((acc0 >> 15), 16);
                    int32_t sat1 = __SSAT((acc1 >> 15), 16);
                    int32_t sat2 = __SSAT((acc2 >> 15), 16);
                    int32_t sat3 = __SSAT((acc3 >> 15), 16);
                    write_q15x2_ia(&pOut, __PKHBT(sat0, sat1, 16));
                    write_q15x2_ia(&pOut, __PKHBT(sat2, sat3, 16));
                }

                /* Increment the pointer pIn1 index, count by 4 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }

            /* If the blockSize2 is not a multiple of 4, compute any remaining output samples here.
               No loop unrolling is used. */
            blkCnt = (uint32_t)blockSize2 & 0x3U;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                   a second loop below computes MACs for the remaining 1 to 3 samples. */
                while (k > 0U)
                {
                    /* Perform the multiply-accumulates */
                    sum += (q63_t)((q31_t)*px++ * *py--);
                    sum += (q63_t)((q31_t)*px++ * *py--);
                    sum += (q63_t)((q31_t)*px++ * *py--);
                    sum += (q63_t)((q31_t)*px++ * *py--);

                    /* Decrement loop counter */
                    k--;
                }

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += (q63_t)((q31_t)*px++ * *py--);

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q15_t)(__SSAT(sum >> 15, 16));

                /* Increment the pointer pIn1 index, count by 1 */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += (q63_t)((q31_t)*px++ * *py--);

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q15_t)(__SSAT(sum >> 15, 16));

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }
#endif /* defined (RISCV_MATH_VECTOR) */

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The count variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;
#if defined(RISCV_MATH_VECTOR)
        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;
        while (blockSize3 > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint16m4_t vx, vy;
            vint32m1_t temp00m1;
            ptrdiff_t bstride = -2;
            l = __riscv_vsetvl_e32m1(vblkCnt);
            temp00m1 = __riscv_vmv_v_x_i32m1(0, l);
            for (; (l = __riscv_vsetvl_e16m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle16_v_i16m4(px, l);
                px += l;
                vy = __riscv_vlse16_v_i16m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i32m8_i32m1(__riscv_vwmul_vv_i32m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i32m1_i32(temp00m1);

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Increment MAC count */
            count--;

            /* Decrement loop counter */
            blockSize3--;
        }
#else
        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        pIn2 = pSrc2 - 1U;
        py = pIn2;

        /* -------------------
         * Stage3 process
         * ------------------*/

        /* For loop unrolling by 4, this stage is divided into two. */
        /* First part of this stage computes the MAC operations greater than 4 */
        /* Second part of this stage computes the MAC operations less than or equal to 4 */

        /* The first part of the stage starts here */
        j = count >> 2U;

        while ((j > 0U) && (blockSize3 > 0))
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count >> 2U;

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
             ** a second loop below computes MACs for the remaining 1 to 3 samples. */
            py = py + 1U;
            while (k > 0U)
            {
                /* x[srcALen - srcBLen + 1], x[srcALen - srcBLen + 2] are multiplied
                 * with y[srcBLen - 1], y[srcBLen - 2] respectively */
                tmp0 = *py--;
                tmp1 = *py--;
                sum = __SMLALD(read_q15x2_ia((q15_t **)&px), __RV_PKBB16(tmp1, tmp0), sum);
                /* x[srcALen - srcBLen + 3], x[srcALen - srcBLen + 4] are multiplied
                 * with y[srcBLen - 3], y[srcBLen - 4] respectively */
                tmp0 = *py--;
                tmp1 = *py--;
                sum = __SMLALD(read_q15x2_ia((q15_t **)&px), __RV_PKBB16(tmp1, tmp0), sum);

                /* Decrement loop counter */
                k--;
            }

            /* If the count is not a multiple of 4, compute any remaining MACs here.
             ** No loop unrolling is used. */
            k = count & 0x3U;

            while (k > 0U)
            {
                /* sum += x[srcALen - srcBLen + 5] * y[srcBLen - 5] */
                sum = __SMLALD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pIn2;

            /* Decrement MAC count */
            count--;

            /* Decrement loop counter */
            blockSize3--;

            j--;
        }

        /* The second part of the stage starts here */
        /* SIMD is not used for the next MAC operations,
         * so pointer py is updated to read only one sample at a time */
        py = py + 1U;

        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum = __SMLALD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((sum >> 15), 16));

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#else /* #if defined (RISCV_MATH_DSP) */

    const q15_t *pIn1 = pSrcA; /* InputA pointer */
    const q15_t *pIn2 = pSrcB; /* InputB pointer */
    q63_t sum;                 /* Accumulator */
    uint32_t i, j;             /* Loop counters */
    riscv_status status;       /* Status of Partial convolution */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* Loop to calculate convolution for output length number of values */
        for (i = firstIndex; i <= (firstIndex + numPoints - 1); i++)
        {
            /* Initialize sum with zero to carry on MAC operations */
            sum = 0;

            /* Loop to perform MAC operations according to convolution equation */
            for (j = 0U; j <= i; j++)
            {
                /* Check the array limitations */
                if (((i - j) < srcBLen) && (j < srcALen))
                {
                    /* z[i] += x[i-j] * y[j] */
                    sum += ((q31_t)pIn1[j] * pIn2[i - j]);
                }
            }

            /* Store the output in the destination buffer */
            pDst[i] = (q15_t)__SSAT((sum >> 15U), 16U);
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#endif /* defined(RISCV_MATH_DSP) || defined (RISCV_MATH_VECTOR) */
}

#define ARRAYA_SIZE_Q31 1024
#define ARRAYB_SIZE_Q31 1024

static q31_t test_conv_input_q31_A[ARRAYA_SIZE_Q31] = {};
static q31_t test_conv_input_q31_B[ARRAYB_SIZE_Q31] = {};

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_q31(
    const q31_t *pSrcA,
    uint32_t srcALen,
    const q31_t *pSrcB,
    uint32_t srcBLen,
    q31_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{

#if defined(RISCV_MATH_DSP) || defined(RISCV_MATH_VECTOR)

    const q31_t *pIn1;          /* InputA pointer */
    const q31_t *pIn2;          /* InputB pointer */
    q31_t *pOut = pDst;         /* Output pointer */
    const q31_t *px;            /* Intermediate inputA pointer */
    const q31_t *py;            /* Intermediate inputB pointer */
    const q31_t *pSrc1, *pSrc2; /* Intermediate pointers */
    q63_t sum;                  /* Accumulator */
    uint32_t j, k, count, blkCnt, check;
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    riscv_status status;                        /* Status of Partial convolution */

#if defined(RISCV_MATH_LOOPUNROLL)
    q63_t acc0, acc1, acc2, acc3; /* Accumulator */
    q31_t x0, x1, x2, x3, c0;     /* Temporary variables */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
    q63_t px64, py64, c064;
    q31_t tmp0, tmp1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
#endif

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = (int32_t)check - ((blockSize3 + blockSize1) + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + firstIndex;
        py = pSrc2;

        /* ------------------------
         * Stage1 process
         * ----------------------*/

        /* The first stage starts here */
        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
#if defined(RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64)
            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint32m4_t vx, vy;
            vint64m1_t temp00m1;
            ptrdiff_t bstride = -4;
            l = __riscv_vsetvl_e64m1(1);
            temp00m1 = __riscv_vmv_v_x_i64m1(0, l);
            for (; (l = __riscv_vsetvl_e32m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_i32m4(px, l);
                px += l;
                vy = __riscv_vlse32_v_i32m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i64m8_i64m1(__riscv_vwmul_vv_i64m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i64m1_i64(temp00m1);
#else
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                px64 = read_q31x2_ia((q31_t **)&px);
                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                sum = __RV_KMADA32(sum, px64, py64);

                px64 = read_q31x2_ia((q31_t **)&px);
                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                sum = __RV_KMADA32(sum, px64, py64);
#else
                /* x[0] * y[srcBLen - 1] */
                sum += (q63_t)*px++ * (*py--);

                /* x[1] * y[srcBLen - 2] */
                sum += (q63_t)*px++ * (*py--);

                /* x[2] * y[srcBLen - 3] */
                sum += (q63_t)*px++ * (*py--);

                /* x[3] * y[srcBLen - 4] */
                sum += (q63_t)*px++ * (*py--);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }
            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;

#else

            /* Initialize k with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                sum += (q63_t)*px++ * (*py--);

                /* Decrement loop counter */
                k--;
            }
#endif /* #if defined (RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64) */
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q31_t)(sum >> 31);

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64)
        blkCnt = blockSize2;

        while (blkCnt > 0U)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;
            uint32_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vint32m4_t vx, vy;
            vint64m1_t temp00m1;
            ptrdiff_t bstride = -4;
            l = __riscv_vsetvl_e64m1(1);
            temp00m1 = __riscv_vmv_v_x_i64m1(0, l);
            for (; (l = __riscv_vsetvl_e32m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_i32m4(px, l);
                px += l;
                vy = __riscv_vlse32_v_i32m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i64m8_i64m1(__riscv_vwmul_vv_i64m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i64m1_i64(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q31_t)(sum >> 31);

            /* Increment MAC count */
            count++;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = pIn1 + count;
            py = pSrc2;

            /* Decrement loop counter */
            blkCnt--;
        }
#else
        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unroll over blkCnt */
            blkCnt = blockSize2 >> 2U;

            while (blkCnt > 0U)
            {
                /* Set all accumulators to zero */
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;
                acc3 = 0;

                /* read x[0], x[1], x[2] samples */
                x0 = *px++;
                x1 = *px++;
                x2 = *px++;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                py -= 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

                /* First part of the processing with loop unrolling.  Compute 3 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 2 samples. */
                do
                {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                    c064 = read_q31x2_da((q31_t **)&py);
                    x3 = *px++;
                    px64 = __RV_PKBB32(x1, x0);
                    acc0 = __RV_KMAXDA32(acc0, px64, c064);
                    px64 = __RV_PKBB32(x2, x1);
                    acc1 = __RV_KMAXDA32(acc1, px64, c064);
                    px64 = __RV_PKBB32(x3, x2);
                    acc2 = __RV_KMAXDA32(acc2, px64, c064);
                    x0 = *px++;
                    px64 = __RV_PKBB32(x0, x3);
                    acc3 = __RV_KMAXDA32(acc3, px64, c064);

                    c064 = read_q31x2_da((q31_t **)&py);
                    px64 = __RV_PKBB32(x3, x2);
                    acc0 = __RV_KMAXDA32(acc0, px64, c064);
                    px64 = __RV_PKBB32(x0, x3);
                    acc1 = __RV_KMAXDA32(acc1, px64, c064);
                    x1 = *px++;
                    px64 = __RV_PKBB32(x1, x0);
                    acc2 = __RV_KMAXDA32(acc2, px64, c064);
                    x2 = *px++;
                    px64 = __RV_PKBB32(x2, x1);
                    acc3 = __RV_KMAXDA32(acc3, px64, c064);
#else
                    /* Read y[srcBLen - 1] sample */
                    c0 = *py--;

                    /* Read x[3] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[0] * y[srcBLen - 1] */
                    acc0 += (q63_t)x0 * c0;
                    /* acc1 +=  x[1] * y[srcBLen - 1] */
                    acc1 += (q63_t)x1 * c0;
                    /* acc2 +=  x[2] * y[srcBLen - 1] */
                    acc2 += (q63_t)x2 * c0;
                    /* acc3 +=  x[3] * y[srcBLen - 1] */
                    acc3 += (q63_t)x3 * c0;

                    /* Read y[srcBLen - 2] sample */
                    c0 = *py--;

                    /* Read x[4] sample */
                    x0 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[1] * y[srcBLen - 2] */
                    acc0 += (q63_t)x1 * c0;
                    /* acc1 +=  x[2] * y[srcBLen - 2] */
                    acc1 += (q63_t)x2 * c0;
                    /* acc2 +=  x[3] * y[srcBLen - 2] */
                    acc2 += (q63_t)x3 * c0;
                    /* acc3 +=  x[4] * y[srcBLen - 2] */
                    acc3 += (q63_t)x0 * c0;

                    /* Read y[srcBLen - 3] sample */
                    c0 = *py--;

                    /* Read x[5] sample */
                    x1 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[2] * y[srcBLen - 3] */
                    acc0 += (q63_t)x2 * c0;
                    /* acc1 +=  x[3] * y[srcBLen - 2] */
                    acc1 += (q63_t)x3 * c0;
                    /* acc2 +=  x[4] * y[srcBLen - 2] */
                    acc2 += (q63_t)x0 * c0;
                    /* acc3 +=  x[4] * y[srcBLen - 2] */
                    acc3 += (q63_t)x1 * c0;

                    /* Read y[srcBLen - 4] sample */
                    c0 = *py--;

                    /* Read x[6] sample */
                    x2 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[2] * y[srcBLen - 3] */
                    acc0 += (q63_t)x3 * c0;
                    /* acc1 +=  x[3] * y[srcBLen - 2] */
                    acc1 += (q63_t)x0 * c0;
                    /* acc2 +=  x[4] * y[srcBLen - 2] */
                    acc2 += (q63_t)x1 * c0;
                    /* acc3 +=  x[4] * y[srcBLen - 2] */
                    acc3 += (q63_t)x2 * c0;

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                } while (--k);

                /* If the srcBLen is not a multiple of 3, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                py += 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

                while (k > 0U)
                {
                    /* Read y[srcBLen - 5] sample */
                    c0 = *py--;
                    /* Read x[7] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[4] * y[srcBLen - 5] */
                    acc0 += (q63_t)x0 * c0;
                    /* acc1 +=  x[5] * y[srcBLen - 5] */
                    acc1 += (q63_t)x1 * c0;
                    /* acc2 +=  x[6] * y[srcBLen - 5] */
                    acc2 += (q63_t)x2 * c0;
                    /* acc3 +=  x[7] * y[srcBLen - 5] */
                    acc3 += (q63_t)x3 * c0;

                    /* Reuse the present samples for the next MAC */
                    x0 = x1;
                    x1 = x2;
                    x2 = x3;

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q31_t)(acc0 >> 31);
                *pOut++ = (q31_t)(acc1 >> 31);
                *pOut++ = (q31_t)(acc2 >> 31);
                *pOut++ = (q31_t)(acc3 >> 31);

                /* Increment the pointer pIn1 index, count by 3 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }

            /* Loop unrolling: Compute remaining outputs */
            blkCnt = blockSize2 & 3U;

#else

            /* Initialize blkCnt with number of samples */
            blkCnt = blockSize2;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

#if defined(RISCV_MATH_LOOPUNROLL)

                /* Loop unrolling: Compute 4 outputs at a time */
                k = srcBLen >> 2U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                py -= 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                while (k > 0U)
                {

                    /* Perform the multiply-accumulates */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                    tmp0 = *px++;
                    tmp1 = *px++;
                    px64 = __RV_PKBB32(tmp1, tmp0);
                    py64 = read_q31x2_da((q31_t **)&py);
                    sum = __RV_KMAXDA32(sum, px64, py64);

                    tmp0 = *px++;
                    tmp1 = *px++;
                    px64 = __RV_PKBB32(tmp1, tmp0);
                    py64 = read_q31x2_da((q31_t **)&py);
                    sum = __RV_KMAXDA32(sum, px64, py64);
#else
                    sum += (q63_t)*px++ * (*py--);
                    sum += (q63_t)*px++ * (*py--);
                    sum += (q63_t)*px++ * (*py--);
                    sum += (q63_t)*px++ * (*py--);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                    /* Decrement loop counter */
                    k--;
                }
                /* Loop unrolling: Compute remaining outputs */
                k = srcBLen & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                py += 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

#else

                /* Initialize blkCnt with number of samples */
                k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += (q63_t)*px++ * *py--;

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q31_t)(sum >> 31);

                /* Increment MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += (q63_t)*px++ * *py--;

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q31_t)(sum >> 31);

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }
#endif /* defined (RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64) */

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The blockSize3 variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* -------------------
         * Stage3 process
         * ------------------*/
#if defined(RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64)
        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            uint32_t vblkCnt = count; /* Loop counter */
            size_t l;
            vint32m4_t vx, vy;
            vint64m1_t temp00m1;
            ptrdiff_t bstride = -4;
            l = __riscv_vsetvl_e64m1(1);
            temp00m1 = __riscv_vmv_v_x_i64m1(0, l);
            for (; (l = __riscv_vsetvl_e32m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle32_v_i32m4(px, l);
                px += l;
                vy = __riscv_vlse32_v_i32m4(py, bstride, l);
                py -= l;
                temp00m1 = __riscv_vredsum_vs_i64m8_i64m1(__riscv_vwmul_vv_i64m8(vx, vy, l), temp00m1, l);
            }
            sum += __riscv_vmv_x_s_i64m1_i64(temp00m1);
            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q31_t)(sum >> 31);

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }
#else
        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

#if defined(RISCV_MATH_LOOPUNROLL)
            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            py -= 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

            while (k > 0U)
            {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                tmp0 = *px++;
                tmp1 = *px++;
                px64 = __RV_PKBB32(tmp1, tmp0);
                py64 = read_q31x2_da((q31_t **)&py);
                sum = __RV_KMAXDA32(sum, px64, py64);
                tmp0 = *px++;
                tmp1 = *px++;
                px64 = __RV_PKBB32(tmp1, tmp0);
                py64 = read_q31x2_da((q31_t **)&py);
                sum = __RV_KMAXDA32(sum, px64, py64);
#else
                /* sum += x[srcALen - srcBLen + 1] * y[srcBLen - 1] */
                sum += (q63_t)*px++ * *py--;

                /* sum += x[srcALen - srcBLen + 2] * y[srcBLen - 2] */
                sum += (q63_t)*px++ * *py--;

                /* sum += x[srcALen - srcBLen + 3] * y[srcBLen - 3] */
                sum += (q63_t)*px++ * *py--;

                /* sum += x[srcALen - srcBLen + 4] * y[srcBLen - 4] */
                sum += (q63_t)*px++ * *py--;

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }
            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            py += 1;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

#else
            /* Initialize blkCnt with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum += (q63_t)*px++ * *py--;

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q31_t)(sum >> 31);

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }
#endif /* defined (RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64) */
        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#else
    /* alternate version for CM0_FAMILY */

    const q31_t *pIn1 = pSrcA; /* InputA pointer */
    const q31_t *pIn2 = pSrcB; /* InputB pointer */
    q63_t sum;                 /* Accumulator */
    uint32_t i, j;             /* Loop counters */
    riscv_status status;       /* Status of Partial convolution */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* Loop to calculate convolution for output length number of values */
        for (i = firstIndex; i <= (firstIndex + numPoints - 1); i++)
        {
            /* Initialize sum with zero to carry on MAC operations */
            sum = 0;

            /* Loop to perform MAC operations according to convolution equation */
            for (j = 0U; j <= i; j++)
            {
                /* Check the array limitations */
                if (((i - j) < srcBLen) && (j < srcALen))
                {
                    /* z[i] += x[i-j] * y[j] */
                    sum += ((q63_t)pIn1[j] * pIn2[i - j]);
                }
            }

            /* Store the output in the destination buffer */
            pDst[i] = (q31_t)(sum >> 31U);
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);

#endif /* defined(RISCV_MATH_DSP) || defined (RISCV_MATH_VECTOR) */
}

__STATIC_FORCEINLINE uint32_t __SMLAD(
    uint32_t x,
    uint32_t y,
    uint32_t sum)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x) >> 16) * (((q31_t)y) >> 16)) +
                       (((q31_t)sum))));
}

/*
 * @brief C custom defined SMLADX
 */
__STATIC_FORCEINLINE uint32_t __SMLADX(
    uint32_t x,
    uint32_t y,
    uint32_t sum)
{
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y) >> 16)) +
                       ((((q31_t)x) >> 16) * (((q31_t)y << 16) >> 16)) +
                       (((q31_t)sum))));
}

__STATIC_FORCEINLINE uint32_t __LW(volatile void *addr)
{
    uint32_t result;

    __ASM volatile("lw %0, 0(%1)" : "=r"(result) : "r"(addr));
    return result;
}

__STATIC_FORCEINLINE q31_t read_q15x2(
    q15_t const *pQ15)
{
    q31_t val;

#ifdef __RISCV_FEATURE_UNALIGNED
    memcpy(&val, pQ15, 4);
#else
    val = __LW((q15_t *)pQ15);
#endif

    return (val);
}

__STATIC_FORCEINLINE q31_t read_q15x2_ia(
    q15_t **pQ15)
{
    q31_t val;

    val = read_q15x2(*pQ15);
    *pQ15 += 2;

    return (val);
}

__STATIC_FORCEINLINE q31_t read_q15x2_da(
    q15_t **pQ15)
{
    q31_t val;

    val = read_q15x2(*pQ15);
    *pQ15 -= 2;

    return (val);
}

__STATIC_FORCEINLINE void __SW(volatile void *addr, uint32_t val)
{
    __ASM volatile("sw %0, 0(%1)" : : "r"(val), "r"(addr));
}

__STATIC_FORCEINLINE void write_q15x2(
    q15_t *pQ15,
    q31_t value)
{
#ifdef __RISCV_FEATURE_UNALIGNED
    memcpy(pQ15, &value, 4);
#else
    __SW(pQ15, value);
#endif
}

__STATIC_FORCEINLINE void write_q15x2_ia(
    q15_t **pQ15,
    q31_t value)
{
    write_q15x2(*pQ15, value);
    *pQ15 += 2;
}

#ifndef RISCV_MATH_DSP
/**
 * @brief definition to pack two 16 bit values.
 */
#define __PKHBT(ARG1, ARG2, ARG3) ((((int32_t)(ARG1) << 0) & (int32_t)0x0000FFFF) | \
                                   (((int32_t)(ARG2) << ARG3) & (int32_t)0xFFFF0000))
#define __PKHTB(ARG1, ARG2, ARG3) ((((int32_t)(ARG1) << 0) & (int32_t)0xFFFF0000) | \
                                   (((int32_t)(ARG2) >> ARG3) & (int32_t)0x0000FFFF))
#endif

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_fast_q15(
    const q15_t *pSrcA,
    uint32_t srcALen,
    const q15_t *pSrcB,
    uint32_t srcBLen,
    q15_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{
#if defined(RISCV_MATH_VECTOR)
    return riscv_conv_partial_q15(pSrcA, srcALen, pSrcB, srcBLen, pDst, firstIndex, numPoints);
#else
    const q15_t *pIn1;                 /* InputA pointer */
    const q15_t *pIn2;                 /* InputB pointer */
    q15_t *pOut = pDst;                /* Output pointer */
    q31_t sum, acc0, acc1, acc2, acc3; /* Accumulator */
    const q15_t *px;                   /* Intermediate inputA pointer */
    const q15_t *py;                   /* Intermediate inputB pointer */
    const q15_t *pSrc1, *pSrc2;        /* Intermediate pointers */
    q31_t x0, x1, x2, x3, c0;          /* Temporary input variables */
    uint32_t j, k, count, blkCnt, check;
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    riscv_status status;                        /* Status of Partial convolution */
    q15_t tmp0, tmp1, tmp2, tmp3;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
    q63_t px64, py64;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = (int32_t)check - ((blockSize3 + blockSize1) + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + firstIndex;
        py = pSrc2;

        /* ------------------------
         * Stage1 process
         * ----------------------*/

        /* For loop unrolling by 4, this stage is divided into two. */
        /* First part of this stage computes the MAC operations less than 4 */
        /* Second part of this stage computes the MAC operations greater than or equal to 4 */

        /* The first part of the stage starts here */
        while ((count < 4U) && (blockSize1 > 0))
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Loop over number of MAC operations between
             * inputA samples and inputB samples */
            k = count;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                sum = __SMLAD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(sum >> 15);

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* The second part of the stage starts here */
        /* The internal loop, over count, is unrolled by 4 */
        /* To, read the last two inputB samples using SIMD:
         * y[srcBLen] and y[srcBLen-1] coefficients, py is decremented by 1 */
#if !(defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64))
        py = py - 1;
#endif /* !(defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64)) */

        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count >> 2U;

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
               a second loop below computes MACs for the remaining 1 to 3 samples. */
            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                px64 = read_q15x4_ia((q15_t **)&px);
                tmp0 = *py--;
                tmp1 = *py--;
                tmp2 = *py--;
                tmp3 = *py--;
                py64 = __RV_PKBB32(__RV_PKBB16(tmp3, tmp2), __RV_PKBB16(tmp1, tmp0));
                sum = __SMLALD(px64, py64, sum);
#else
                /* x[0], x[1] are multiplied with y[srcBLen - 1], y[srcBLen - 2] respectively */
                sum = __SMLADX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);
                /* x[2], x[3] are multiplied with y[srcBLen - 3], y[srcBLen - 4] respectively */
                sum = __SMLADX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }

            /* For the next MAC operations, the pointer py is used without SIMD
               So, py is incremented by 1 */
#if !(defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64))
            py = py + 1;
#endif /* !(defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64)) */

            /* If the count is not a multiple of 4, compute any remaining MACs here.
               No loop unrolling is used. */
            k = count & 0x3U;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                sum = __SMLAD(*px++, *py--, sum);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(sum >> 15);

            /* Update the inputA and inputB pointers for next MAC calculation */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            py = ++pSrc2;
#else
            py = ++pSrc2 - 1U;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is the index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/

        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
            /* Loop unrolling: Compute 4 outputs at a time */
            blkCnt = ((uint32_t)blockSize2 >> 2U);

            while (blkCnt > 0U)
            {
                py = py - 1U;

                /* Set all accumulators to zero */
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;
                acc3 = 0;

                /* read x[0], x[1] samples */
                x0 = read_q15x2((q15_t *)px);
                /* read x[1], x[2] samples */
#if defined(RISCV_MATH_DSP)
                x1 = __RV_PKBT16(*(px + 2), x0);
#else
                x1 = read_q15x2((q15_t *)px + 1);
#endif /* defined (RISCV_MATH_DSP) */
                px += 2U;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 3 samples. */
                do
                {
                    /* Read the last two inputB samples using SIMD:
                     * y[srcBLen - 1] and y[srcBLen - 2] */
                    c0 = read_q15x2_da((q15_t **)&py);

                    /* acc0 +=  x[0] * y[srcBLen - 1] + x[1] * y[srcBLen - 2] */
                    acc0 = __SMLADX(x0, c0, acc0);

                    /* acc1 +=  x[1] * y[srcBLen - 1] + x[2] * y[srcBLen - 2] */
                    acc1 = __SMLADX(x1, c0, acc1);

                    /* Read x[2], x[3] */
                    x2 = read_q15x2((q15_t *)px);

                    /* Read x[3], x[4] */
#if defined(RISCV_MATH_DSP)
                    x3 = __RV_PKBT16(*(px + 2), x2);
#else
                    x3 = read_q15x2((q15_t *)px + 1);
#endif /* defined (RISCV_MATH_DSP) */

                    /* acc2 +=  x[2] * y[srcBLen - 1] + x[3] * y[srcBLen - 2] */
                    acc2 = __SMLADX(x2, c0, acc2);

                    /* acc3 +=  x[3] * y[srcBLen - 1] + x[4] * y[srcBLen - 2] */
                    acc3 = __SMLADX(x3, c0, acc3);

                    /* Read y[srcBLen - 3] and y[srcBLen - 4] */
                    c0 = read_q15x2_da((q15_t **)&py);

                    /* acc0 +=  x[2] * y[srcBLen - 3] + x[3] * y[srcBLen - 4] */
                    acc0 = __SMLADX(x2, c0, acc0);

                    /* acc1 +=  x[3] * y[srcBLen - 3] + x[4] * y[srcBLen - 4] */
                    acc1 = __SMLADX(x3, c0, acc1);

                    /* Read x[4], x[5] */
                    x0 = read_q15x2((q15_t *)px + 2);

                    /* Read x[5], x[6] */
#if defined(RISCV_MATH_DSP)
                    x1 = __RV_PKBT16(*(px + 4), x0);
#else
                    x1 = read_q15x2((q15_t *)px + 3);
#endif /* defined (RISCV_MATH_DSP) */
                    px += 4U;

                    /* acc2 +=  x[4] * y[srcBLen - 3] + x[5] * y[srcBLen - 4] */
                    acc2 = __SMLADX(x0, c0, acc2);

                    /* acc3 +=  x[5] * y[srcBLen - 3] + x[6] * y[srcBLen - 4] */
                    acc3 = __SMLADX(x1, c0, acc3);

                } while (--k);

                /* For the next MAC operations, SIMD is not used
                   So, the 16 bit pointer if inputB, py is updated */

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                   No loop unrolling is used. */
                k = srcBLen % 0x4U;

                if (k == 1U)
                {
                    /* Read y[srcBLen - 5] */
                    c0 = *(py + 1);
                    c0 = c0 & 0x0000FFFF;

                    /* Read x[7] */
                    x3 = read_q15x2((q15_t *)px);
                    px++;

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLAD(x0, c0, acc0);
                    acc1 = __SMLAD(x1, c0, acc1);
                    acc2 = __SMLADX(x1, c0, acc2);
                    acc3 = __SMLADX(x3, c0, acc3);
                }

                if (k == 2U)
                {
                    /* Read y[srcBLen - 5], y[srcBLen - 6] */
                    c0 = read_q15x2((q15_t *)py);

                    /* Read x[7], x[8] */
                    x3 = read_q15x2((q15_t *)px);

                    /* Read x[9] */
#if defined(RISCV_MATH_DSP)
                    x2 = __RV_PKBT16(*(px + 2), x3);
#else
                    x2 = read_q15x2((q15_t *)px + 1);
#endif /* defined (RISCV_MATH_DSP) */
                    px += 2U;

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLADX(x0, c0, acc0);
                    acc1 = __SMLADX(x1, c0, acc1);
                    acc2 = __SMLADX(x3, c0, acc2);
                    acc3 = __SMLADX(x2, c0, acc3);
                }

                if (k == 3U)
                {
                    /* Read y[srcBLen - 5], y[srcBLen - 6] */
                    c0 = read_q15x2((q15_t *)py);

                    /* Read x[7], x[8] */
                    x3 = read_q15x2((q15_t *)px);

                    /* Read x[9] */
#if defined(RISCV_MATH_DSP)
                    x2 = __RV_PKBT16(*(px + 2), x3);
#else
                    x2 = read_q15x2((q15_t *)px + 1);
#endif /* defined (RISCV_MATH_DSP) */

                    /* Perform the multiply-accumulate */
                    acc0 = __SMLADX(x0, c0, acc0);
                    acc1 = __SMLADX(x1, c0, acc1);
                    acc2 = __SMLADX(x3, c0, acc2);
                    acc3 = __SMLADX(x2, c0, acc3);

                    c0 = *(py - 1);
                    c0 = c0 & 0x0000FFFF;

                    /* Read x[10] */
                    x3 = read_q15x2((q15_t *)px + 2);
                    px += 3U;

                    /* Perform the multiply-accumulates */
                    acc0 = __SMLADX(x1, c0, acc0);
                    acc1 = __SMLAD(x2, c0, acc1);
                    acc2 = __SMLADX(x2, c0, acc2);
                    acc3 = __SMLADX(x3, c0, acc3);
                }

                /* Store the results in the accumulators in the destination buffer. */
                write_q15x2_ia(&pOut, __PKHBT(acc0 >> 15, acc1 >> 15, 16));
                write_q15x2_ia(&pOut, __PKHBT(acc2 >> 15, acc3 >> 15, 16));

                /* Increment the pointer pIn1 index, count by 4 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }

            /* If the blockSize2 is not a multiple of 4, compute any remaining output samples here.
               No loop unrolling is used. */
            blkCnt = (unsigned long)blockSize2 & 0x3U;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                   a second loop below computes MACs for the remaining 1 to 3 samples. */
                while (k > 0U)
                {
                    /* Perform the multiply-accumulates */
                    sum += ((q31_t)*px++ * *py--);
                    sum += ((q31_t)*px++ * *py--);
                    sum += ((q31_t)*px++ * *py--);
                    sum += ((q31_t)*px++ * *py--);

                    /* Decrement loop counter */
                    k--;
                }

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulates */
                    sum += ((q31_t)*px++ * *py--);

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q15_t)(sum >> 15);

                /* Increment the pointer pIn1 index, count by 1 */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum += ((q31_t)*px++ * *py--);

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q15_t)(sum >> 15);

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The count variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        pIn2 = pSrc2 - 1U;
        py = pIn2;

        /* -------------------
         * Stage3 process
         * ------------------*/

        /* For loop unrolling by 4, this stage is divided into two. */
        /* First part of this stage computes the MAC operations greater than 4 */
        /* Second part of this stage computes the MAC operations less than or equal to 4 */

        /* The first part of the stage starts here */
        j = count >> 2U;

        while ((j > 0U) && (blockSize3 > 0))
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count >> 2U;

            /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
             ** a second loop below computes MACs for the remaining 1 to 3 samples. */
#if defined(RISCV_MATH_DSP)
            py = py + 1U;
#endif /* define (RISCV_MATH_DSP) */
            while (k > 0U)
            {
                /* x[srcALen - srcBLen + 1], x[srcALen - srcBLen + 2] are multiplied
                 * with y[srcBLen - 1], y[srcBLen - 2] respectively */
#if defined(RISCV_MATH_DSP)
                tmp0 = *py--;
                tmp1 = *py--;
                sum = __SMLALD(read_q15x2_ia((q15_t **)&px), __RV_PKBB16(tmp1, tmp0), sum);
                /* x[srcALen - srcBLen + 3], x[srcALen - srcBLen + 4] are multiplied
                 * with y[srcBLen - 3], y[srcBLen - 4] respectively */
                tmp0 = *py--;
                tmp1 = *py--;
                sum = __SMLALD(read_q15x2_ia((q15_t **)&px), __RV_PKBB16(tmp1, tmp0), sum);
#else
                /* x[srcALen - srcBLen + 1], x[srcALen - srcBLen + 2] are multiplied
                 * with y[srcBLen - 1], y[srcBLen - 2] respectively */
                sum = __SMLADX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);
                /* x[srcALen - srcBLen + 3], x[srcALen - srcBLen + 4] are multiplied
                 * with y[srcBLen - 3], y[srcBLen - 4] respectively */
                sum = __SMLADX(read_q15x2_ia((q15_t **)&px), read_q15x2_da((q15_t **)&py), sum);
#endif /* defined (RISCV_MATH_DSP) */

                /* Decrement loop counter */
                k--;
            }

#if !(defined(RISCV_MATH_DSP))
            /* For the next MAC operations, the pointer py is used without SIMD
               So, py is incremented by 1 */
            py = py + 1U;
#endif /* !(defined (RISCV_MATH_DSP)) */

            /* If the count is not a multiple of 4, compute any remaining MACs here.
               No loop unrolling is used. */
            k = count & 0x3U;

            while (k > 0U)
            {
                /* sum += x[srcALen - srcBLen + 5] * y[srcBLen - 5] */
                sum = __SMLAD(*px++, *py--, sum);

                /* Decrement the loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(sum >> 15);

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pIn2;

            /* Decrement the MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;

            j--;
        }

        /* The second part of the stage starts here */
        /* SIMD is not used for the next MAC operations,
         * so pointer py is updated to read only one sample at a time */
        py = py + 1U;

        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
            sum = 0;

            /* Apply loop unrolling and compute 4 MACs simultaneously. */
            k = count;

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum = __SMLAD(*px++, *py--, sum);

                /* Decrement the loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q15_t)(sum >> 15);

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement the MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);
#endif /*defined (RISCV_MATH_VECTOR)*/
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_fast_q31(
    const q31_t *pSrcA,
    uint32_t srcALen,
    const q31_t *pSrcB,
    uint32_t srcBLen,
    q31_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints)
{
    const q31_t *pIn1;          /* InputA pointer */
    const q31_t *pIn2;          /* InputB pointer */
    q31_t *pOut = pDst;         /* Output pointer */
    const q31_t *px;            /* Intermediate inputA pointer */
    const q31_t *py;            /* Intermediate inputB pointer */
    const q31_t *pSrc1, *pSrc2; /* Intermediate pointers */
    q31_t sum;                  /* Accumulators */
    unsigned long j, k, count, check, blkCnt;
    int32_t blockSize1, blockSize2, blockSize3; /* Loop counters */
    riscv_status status;                        /* Status of Partial convolution */

    q31_t acc0, acc1, acc2, acc3; /* Accumulators */
    q31_t x0, x1, x2, x3, c0;
#if defined(RISCV_MATH_LOOPUNROLL)
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
    q31_t tmp0, tmp1;
    q63_t px64, py64, sum64;
    q63_t acc064, acc164, acc264, acc364;
    q63_t x064, x164, x264, x364, c064;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Conditions to check which loopCounter holds
         * the first and last indices of the output samples to be calculated. */
        check = firstIndex + numPoints;
        blockSize3 = ((int32_t)check > (int32_t)srcALen) ? (int32_t)check - (int32_t)srcALen : 0;
        blockSize3 = ((int32_t)firstIndex > (int32_t)srcALen - 1) ? blockSize3 - (int32_t)firstIndex + (int32_t)srcALen : blockSize3;
        blockSize1 = ((int32_t)srcBLen - 1) - (int32_t)firstIndex;
        blockSize1 = (blockSize1 > 0) ? ((check > (srcBLen - 1U)) ? blockSize1 : (int32_t)numPoints) : 0;
        blockSize2 = (int32_t)check - ((blockSize3 + blockSize1) + (int32_t)firstIndex);
        blockSize2 = (blockSize2 > 0) ? blockSize2 : 0;

        /* conv(x,y) at n = x[n] * y[0] + x[n-1] * y[1] + x[n-2] * y[2] + ...+ x[n-N+1] * y[N -1] */
        /* The function is internally
         * divided into three stages according to the number of multiplications that has to be
         * taken place between inputA samples and inputB samples. In the first stage of the
         * algorithm, the multiplications increase by one for every iteration.
         * In the second stage of the algorithm, srcBLen number of multiplications are done.
         * In the third stage of the algorithm, the multiplications decrease by one
         * for every iteration. */

        /* Set the output pointer to point to the firstIndex
         * of the output sample to be calculated. */
        pOut = pDst + firstIndex;

        /* --------------------------
         * Initializations of stage1
         * -------------------------*/

        /* sum = x[0] * y[0]
         * sum = x[0] * y[1] + x[1] * y[0]
         * ....
         * sum = x[0] * y[srcBlen - 1] + x[1] * y[srcBlen - 2] +...+ x[srcBLen - 1] * y[0]
         */

        /* In this stage the MAC operations are increased by 1 for every iteration.
           The count variable holds the number of MAC operations performed.
           Since the partial convolution starts from firstIndex
           Number of Macs to be performed is firstIndex + 1 */
        count = 1U + firstIndex;

        /* Working pointer of inputA */
        px = pIn1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + firstIndex;
        py = pSrc2;

        /* ------------------------
         * Stage1 process
         * ----------------------*/

        /* The first stage starts here */
        while (blockSize1 > 0)
        {
            /* Accumulator is made zero for every iteration */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined(RISCV_MATH_LOOPUNROLL)
            sum64 = 0;
#else
            sum = 0;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined (RISCV_MATH_LOOPUNROLL) */

#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                px64 = read_q31x2_ia((q31_t **)&px);
                sum64 = __RV_KMADA32(sum64, px64, py64);

                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                px64 = read_q31x2_ia((q31_t **)&px);
                sum64 = __RV_KMADA32(sum64, px64, py64);
#else
                /* x[0] * y[srcBLen - 1] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* x[1] * y[srcBLen - 2] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* x[2] * y[srcBLen - 3] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* x[3] * y[srcBLen - 4] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            sum = (q31_t)(sum64 >> 32);
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

#else

            /* Initialize k with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulate */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = sum << 1;

            /* Update the inputA and inputB pointers for next MAC calculation */
            py = ++pSrc2;
            px = pIn1;

            /* Increment MAC count */
            count++;

            /* Decrement loop counter */
            blockSize1--;
        }

        /* --------------------------
         * Initializations of stage2
         * ------------------------*/

        /* sum = x[0] * y[srcBLen-1] + x[1] * y[srcBLen-2] +...+ x[srcBLen-1] * y[0]
         * sum = x[1] * y[srcBLen-1] + x[2] * y[srcBLen-2] +...+ x[srcBLen] * y[0]
         * ....
         * sum = x[srcALen-srcBLen-2] * y[srcBLen-1] + x[srcALen] * y[srcBLen-2] +...+ x[srcALen-1] * y[0]
         */

        /* Working pointer of inputA */
        if ((int32_t)firstIndex - (int32_t)srcBLen + 1 > 0)
        {
            pSrc1 = pIn1 + firstIndex - srcBLen + 1;
        }
        else
        {
            pSrc1 = pIn1;
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* count is index by which the pointer pIn1 to be incremented */
        count = 0U;

        /* -------------------
         * Stage2 process
         * ------------------*/

        /* Stage2 depends on srcBLen as in this stage srcBLen number of MACS are performed.
         * So, to loop unroll over blockSize2,
         * srcBLen should be greater than or equal to 4 */
        if (srcBLen >= 4U)
        {
#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            blkCnt = ((unsigned long)blockSize2 >> 2U);

            while (blkCnt > 0U)
            {
                /* Set all accumulators to zero */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                acc064 = 0;
                acc164 = 0;
                acc264 = 0;
                acc364 = 0;
#else
                acc0 = 0;
                acc1 = 0;
                acc2 = 0;
                acc3 = 0;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

                /* read x[0], x[1], x[2] samples */
                x0 = *px++;
                x1 = *px++;
                x2 = *px++;

                /* Apply loop unrolling and compute 4 MACs simultaneously. */
                k = srcBLen >> 2U;

                /* First part of the processing with loop unrolling.  Compute 4 MACs at a time.
                 ** a second loop below computes MACs for the remaining 1 to 3 samples. */
                do
                {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                    tmp0 = *py--;
                    tmp1 = *py--;
                    c064 = __RV_PKBB32(tmp1, tmp0);

                    x064 = __RV_PKBB32(x1, x0);
                    acc064 = __RV_KMADA32(acc064, x064, c064);

                    x164 = __RV_PKBB32(x2, x1);
                    acc164 = __RV_KMADA32(acc164, x164, c064);

                    /* Read x[3] sample */
                    x3 = *px++;

                    x264 = __RV_PKBB32(x3, x2);
                    acc264 = __RV_KMADA32(acc264, x264, c064);

                    /* Read x[4] sample */
                    x0 = *px++;

                    x364 = __RV_PKBB32(x0, x3);
                    acc364 = __RV_KMADA32(acc364, x364, c064);

                    tmp0 = *py--;
                    tmp1 = *py--;
                    c064 = __RV_PKBB32(tmp1, tmp0);

                    x064 = __RV_PKBB32(x3, x2);
                    acc064 = __RV_KMADA32(acc064, x064, c064);

                    x164 = __RV_PKBB32(x0, x3);
                    acc164 = __RV_KMADA32(acc164, x164, c064);

                    /* Read x[5] sample */
                    x1 = *px++;

                    x264 = __RV_PKBB32(x1, x0);
                    acc264 = __RV_KMADA32(acc264, x264, c064);

                    /* Read x[6] sample */
                    x2 = *px++;

                    x364 = __RV_PKBB32(x2, x1);
                    acc364 = __RV_KMADA32(acc364, x364, c064);
#else
                    /* Read y[srcBLen - 1] sample */
                    c0 = *py--;
                    /* Read x[3] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[0] * y[srcBLen - 1] */

                    acc0 = (q31_t)((((q63_t)acc0 << 32) + ((q63_t)x0 * c0)) >> 32);
                    /* acc1 +=  x[1] * y[srcBLen - 1] */
                    acc1 = (q31_t)((((q63_t)acc1 << 32) + ((q63_t)x1 * c0)) >> 32);
                    /* acc2 +=  x[2] * y[srcBLen - 1] */
                    acc2 = (q31_t)((((q63_t)acc2 << 32) + ((q63_t)x2 * c0)) >> 32);
                    /* acc3 +=  x[3] * y[srcBLen - 1] */
                    acc3 = (q31_t)((((q63_t)acc3 << 32) + ((q63_t)x3 * c0)) >> 32);

                    /* Read y[srcBLen - 2] sample */
                    c0 = *py--;
                    /* Read x[4] sample */
                    x0 = *px++;

                    /* Perform the multiply-accumulate */
                    /* acc0 +=  x[1] * y[srcBLen - 2] */
                    acc0 = (q31_t)((((q63_t)acc0 << 32) + ((q63_t)x1 * c0)) >> 32);
                    /* acc1 +=  x[2] * y[srcBLen - 2] */
                    acc1 = (q31_t)((((q63_t)acc1 << 32) + ((q63_t)x2 * c0)) >> 32);
                    /* acc2 +=  x[3] * y[srcBLen - 2] */
                    acc2 = (q31_t)((((q63_t)acc2 << 32) + ((q63_t)x3 * c0)) >> 32);
                    /* acc3 +=  x[4] * y[srcBLen - 2] */
                    acc3 = (q31_t)((((q63_t)acc3 << 32) + ((q63_t)x0 * c0)) >> 32);

                    /* Read y[srcBLen - 3] sample */
                    c0 = *py--;
                    /* Read x[5] sample */
                    x1 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[2] * y[srcBLen - 3] */
                    acc0 = (q31_t)((((q63_t)acc0 << 32) + ((q63_t)x2 * c0)) >> 32);
                    /* acc1 +=  x[3] * y[srcBLen - 2] */
                    acc1 = (q31_t)((((q63_t)acc1 << 32) + ((q63_t)x3 * c0)) >> 32);
                    /* acc2 +=  x[4] * y[srcBLen - 2] */
                    acc2 = (q31_t)((((q63_t)acc2 << 32) + ((q63_t)x0 * c0)) >> 32);
                    /* acc3 +=  x[5] * y[srcBLen - 2] */
                    acc3 = (q31_t)((((q63_t)acc3 << 32) + ((q63_t)x1 * c0)) >> 32);

                    /* Read y[srcBLen - 4] sample */
                    c0 = *py--;
                    /* Read x[6] sample */
                    x2 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[3] * y[srcBLen - 4] */
                    acc0 = (q31_t)((((q63_t)acc0 << 32) + ((q63_t)x3 * c0)) >> 32);
                    /* acc1 +=  x[4] * y[srcBLen - 4] */
                    acc1 = (q31_t)((((q63_t)acc1 << 32) + ((q63_t)x0 * c0)) >> 32);
                    /* acc2 +=  x[5] * y[srcBLen - 4] */
                    acc2 = (q31_t)((((q63_t)acc2 << 32) + ((q63_t)x1 * c0)) >> 32);
                    /* acc3 +=  x[6] * y[srcBLen - 4] */
                    acc3 = (q31_t)((((q63_t)acc3 << 32) + ((q63_t)x2 * c0)) >> 32);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                } while (--k);

                /* If the srcBLen is not a multiple of 4, compute any remaining MACs here.
                 ** No loop unrolling is used. */
                k = srcBLen & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                acc0 = (q31_t)(acc064 >> 32);
                acc1 = (q31_t)(acc164 >> 32);
                acc2 = (q31_t)(acc264 >> 32);
                acc3 = (q31_t)(acc364 >> 32);
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

                while (k > 0U)
                {
                    /* Read y[srcBLen - 5] sample */
                    c0 = *py--;
                    /* Read x[7] sample */
                    x3 = *px++;

                    /* Perform the multiply-accumulates */
                    /* acc0 +=  x[4] * y[srcBLen - 5] */
                    acc0 = (q31_t)((((q63_t)acc0 << 32) + ((q63_t)x0 * c0)) >> 32);
                    /* acc1 +=  x[5] * y[srcBLen - 5] */
                    acc1 = (q31_t)((((q63_t)acc1 << 32) + ((q63_t)x1 * c0)) >> 32);
                    /* acc2 +=  x[6] * y[srcBLen - 5] */
                    acc2 = (q31_t)((((q63_t)acc2 << 32) + ((q63_t)x2 * c0)) >> 32);
                    /* acc3 +=  x[7] * y[srcBLen - 5] */
                    acc3 = (q31_t)((((q63_t)acc3 << 32) + ((q63_t)x3 * c0)) >> 32);

                    /* Reuse the present samples for the next MAC */
                    x0 = x1;
                    x1 = x2;
                    x2 = x3;

                    /* Decrement the loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = (q31_t)(acc0 << 1);
                *pOut++ = (q31_t)(acc1 << 1);
                *pOut++ = (q31_t)(acc2 << 1);
                *pOut++ = (q31_t)(acc3 << 1);

                /* Increment the pointer pIn1 index, count by 4 */
                count += 4U;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }

            /* Loop unrolling: Compute remaining outputs */
            blkCnt = (uint32_t)blockSize2 & 0x3U;

#else

            /* Initialize blkCnt with number of samples */
            blkCnt = blockSize2;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined(RISCV_MATH_LOOPUNROLL)
                sum64 = 0;
#else
                sum = 0;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined (RISCV_MATH_LOOPUNROLL) */

#if defined(RISCV_MATH_LOOPUNROLL)

                /* Loop unrolling: Compute 4 outputs at a time */
                k = srcBLen >> 2U;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulates */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                    tmp0 = *py--;
                    tmp1 = *py--;
                    py64 = __RV_PKBB32(tmp1, tmp0);
                    tmp0 = *px++;
                    tmp1 = *px++;
                    px64 = __RV_PKBB32(tmp1, tmp0);
                    sum64 = __RV_KMADA32(sum64, px64, py64);

                    tmp0 = *py--;
                    tmp1 = *py--;
                    py64 = __RV_PKBB32(tmp1, tmp0);
                    tmp0 = *px++;
                    tmp1 = *px++;
                    px64 = __RV_PKBB32(tmp1, tmp0);
                    sum64 = __RV_KMADA32(sum64, px64, py64);
#else
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                    /* Decrement loop counter */
                    k--;
                }

                /* Loop unrolling: Compute remaining outputs */
                k = srcBLen & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                sum = (q31_t)(sum64 >> 32);
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

#else

                /* Initialize blkCnt with number of samples */
                k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = sum << 1;

                /* Increment MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement loop counter */
                blkCnt--;
            }
        }
        else
        {
            /* If the srcBLen is not a multiple of 4,
             * the blockSize2 loop cannot be unrolled by 4 */
            blkCnt = (uint32_t)blockSize2;

            while (blkCnt > 0U)
            {
                /* Accumulator is made zero for every iteration */
                sum = 0;

                /* srcBLen number of MACS should be performed */
                k = srcBLen;

                while (k > 0U)
                {
                    /* Perform the multiply-accumulate */
                    sum = (q31_t)((((q63_t)sum << 32) +
                                   ((q63_t)*px++ * (*py--))) >>
                                  32);

                    /* Decrement loop counter */
                    k--;
                }

                /* Store the result in the accumulator in the destination buffer. */
                *pOut++ = sum << 1;

                /* Increment the MAC count */
                count++;

                /* Update the inputA and inputB pointers for next MAC calculation */
                px = pSrc1 + count;
                py = pSrc2;

                /* Decrement the loop counter */
                blkCnt--;
            }
        }

        /* --------------------------
         * Initializations of stage3
         * -------------------------*/

        /* sum += x[srcALen-srcBLen+1] * y[srcBLen-1] + x[srcALen-srcBLen+2] * y[srcBLen-2] +...+ x[srcALen-1] * y[1]
         * sum += x[srcALen-srcBLen+2] * y[srcBLen-1] + x[srcALen-srcBLen+3] * y[srcBLen-2] +...+ x[srcALen-1] * y[2]
         * ....
         * sum +=  x[srcALen-2] * y[srcBLen-1] + x[srcALen-1] * y[srcBLen-2]
         * sum +=  x[srcALen-1] * y[srcBLen-1]
         */

        /* In this stage the MAC operations are decreased by 1 for every iteration.
           The count variable holds the number of MAC operations performed */
        count = srcBLen - 1U;

        /* Working pointer of inputA */
        if (firstIndex > srcALen)
        {
            pSrc1 = (pIn1 + firstIndex) - (srcBLen - 1U);
        }
        else
        {
            pSrc1 = (pIn1 + srcALen) - (srcBLen - 1U);
        }
        px = pSrc1;

        /* Working pointer of inputB */
        pSrc2 = pIn2 + (srcBLen - 1U);
        py = pSrc2;

        /* -------------------
         * Stage3 process
         * ------------------*/

        while (blockSize3 > 0)
        {
            /* Accumulator is made zero for every iteration */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined(RISCV_MATH_LOOPUNROLL)
            sum64 = 0;
#else
            sum = 0;
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) && defined (RISCV_MATH_LOOPUNROLL) */

#if defined(RISCV_MATH_LOOPUNROLL)

            /* Loop unrolling: Compute 4 outputs at a time */
            k = count >> 2U;

            while (k > 0U)
            {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                tmp0 = *px++;
                tmp1 = *px++;
                px64 = __RV_PKBB32(tmp1, tmp0);
                sum64 = __RV_KMADA32(sum64, px64, py64);

                tmp0 = *py--;
                tmp1 = *py--;
                py64 = __RV_PKBB32(tmp1, tmp0);
                tmp0 = *px++;
                tmp1 = *px++;
                px64 = __RV_PKBB32(tmp1, tmp0);
                sum64 = __RV_KMADA32(sum64, px64, py64);
#else
                /* sum += x[srcALen - srcBLen + 1] * y[srcBLen - 1] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* sum += x[srcALen - srcBLen + 2] * y[srcBLen - 2] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* sum += x[srcALen - srcBLen + 3] * y[srcBLen - 3] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* sum += x[srcALen - srcBLen + 4] * y[srcBLen - 4] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
                /* Decrement loop counter */
                k--;
            }

            /* Loop unrolling: Compute remaining outputs */
            k = count & 0x3U;
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            sum = (q31_t)(sum64 >> 32);
#endif /* defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

#else

            /* Initialize blkCnt with number of samples */
            k = count;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

            while (k > 0U)
            {
                /* Perform the multiply-accumulates */
                /* sum +=  x[srcALen-1] * y[srcBLen-1] */
                sum = (q31_t)((((q63_t)sum << 32) +
                               ((q63_t)*px++ * (*py--))) >>
                              32);

                /* Decrement loop counter */
                k--;
            }

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = sum << 1;

            /* Update the inputA and inputB pointers for next MAC calculation */
            px = ++pSrc1;
            py = pSrc2;

            /* Decrement MAC count */
            count--;

            /* Decrement the loop counter */
            blockSize3--;
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);
}

#define ARRAYA_SIZE_Q7 1024
#define ARRAYB_SIZE_Q7 1024

static q7_t test_conv_input_q7_A[ARRAYA_SIZE_Q7];
static q7_t test_conv_input_q7_B[ARRAYB_SIZE_Q7];
static q15_t pScratch1[max(ARRAYA_SIZE_Q7, ARRAYB_SIZE_Q7) + 2 * min(ARRAYA_SIZE_Q7, ARRAYB_SIZE_Q7) - 2];
static q15_t pScratch2[min(ARRAYA_SIZE_Q7, ARRAYB_SIZE_Q7)];

RISCV_DSP_ATTRIBUTE void riscv_fill_q15(
    q15_t value,
    q15_t *pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt; /* Loop counter */

#if defined(RISCV_MATH_VECTOR)
    blkCnt = blockSize; /* Loop counter */
    size_t l;
    vint16m8_t v_fill;
    l = __riscv_vsetvlmax_e16m8();
    v_fill = __riscv_vmv_v_x_i16m8(value, l);
    for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l)
    {
        __riscv_vse16_v_i16m8(pDst, v_fill, l);
        pDst += l;
    }
#else
#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t packedValue; /* value packed to 32 bits */

    /* Packing two 16 bit values to 32 bit value in order to use SIMD */
    // packedValue = __PKHBT(value, value, 16U);
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
    q63_t packedValue64; /* value packed to 64 bits */
    packedValue = __PKBB16(value, value);
    packedValue64 = __RV_PKBB32(packedValue, packedValue);
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* C = value */

        /* fill 4 times 2 samples at a time */
        write_q15x4_ia(&pDst, packedValue64);
        blkCnt--;
    }
#else
    packedValue = __PKHBT(value, value, 16U);
    blkCnt = blockSize >> 2U;
    while (blkCnt > 0U)
    {
        /* C = value */

        /* fill 2 times 2 samples at a time */
        write_q15x2_ia(&pDst, packedValue);
        write_q15x2_ia(&pDst, packedValue);
        blkCnt--;
    }
#endif /* RISCV_MATH_DSP && __RISCV_XLEN == 64 */

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* C = value */

        /* Fill value in destination buffer */
        *pDst++ = value;

        /* Decrement loop counter */
        blkCnt--;
    }
#endif /* defined(RISCV_MATH_VECTOR) */
}

__STATIC_FORCEINLINE void write_q7x4(
    q7_t *pQ7,
    q31_t value)
{
    q31_t val = value;
#ifdef __RISCV_FEATURE_UNALIGNED
    memcpy(pQ7, &val, 4);
#else
    __SW(pQ7, val);
#endif
}

__STATIC_FORCEINLINE void write_q7x4_ia(
    q7_t **pQ7,
    q31_t value)
{
    write_q7x4(*pQ7, value);
    *pQ7 += 4;
}

#define __PACKq7(v0, v1, v2, v3) ((((int32_t)(v0) << 0) & (int32_t)0x000000FF) |  \
                                  (((int32_t)(v1) << 8) & (int32_t)0x0000FF00) |  \
                                  (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) | \
                                  (((int32_t)(v3) << 24) & (int32_t)0xFF000000))

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_opt_q7(
    const q7_t *pSrcA,
    uint32_t srcALen,
    const q7_t *pSrcB,
    uint32_t srcBLen,
    q7_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints,
    q15_t *pScratch1,
    q15_t *pScratch2)
{
    q15_t *pScr2, *pScr1;          /* Intermediate pointers for scratch pointers */
    q15_t x4;                      /* Temporary input variable */
    const q7_t *pIn1, *pIn2;       /* InputA and inputB pointer */
    uint32_t j, k, blkCnt, tapCnt; /* Loop counter */
    const q7_t *px;                /* Temporary input1 pointer */
    q15_t *py;                     /* Temporary input2 pointer */
    q31_t acc0, acc1, acc2, acc3;  /* Accumulator */
    q31_t x1, x2, x3, y1;          /* Temporary input variables */
    riscv_status status;
    q7_t *pOut = pDst;           /* Output pointer */
    q7_t out0, out1, out2, out3; /* Temporary variables */

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* pointer to take end of scratch2 buffer */
        pScr2 = pScratch2;

        /* points to smaller length sequence */
        px = pIn2 + srcBLen - 1;
#if defined(RISCV_MATH_VECTOR)
        uint32_t vblkCnt = srcBLen; /* Loop counter */
        size_t l;
        vint16m8_t vx;
        ptrdiff_t bstride = -1;
        for (; (l = __riscv_vsetvl_e8m4(vblkCnt)) > 0; vblkCnt -= l)
        {
            vx = __riscv_vwadd_vx_i16m8(__riscv_vlse8_v_i8m4(px, bstride, l), 0, l);
            px -= l;
            __riscv_vse16_v_i16m8(pScr2, vx, l);
            pScr2 += l;
        }
#else
        /* Apply loop unrolling and do 4 Copies simultaneously. */
        k = srcBLen >> 2U;

        /* First part of the processing with loop unrolling copies 4 data points at a time.
         ** a second loop below copies for the remaining 1 to 3 samples. */
        while (k > 0U)
        {
            /* copy second buffer in reversal manner */
            x4 = (q15_t)*px--;
            *pScr2++ = x4;
            x4 = (q15_t)*px--;
            *pScr2++ = x4;
            x4 = (q15_t)*px--;
            *pScr2++ = x4;
            x4 = (q15_t)*px--;
            *pScr2++ = x4;

            /* Decrement loop counter */
            k--;
        }

        /* If the count is not a multiple of 4, copy remaining samples here.
         ** No loop unrolling is used. */
        k = srcBLen & 0x3U;

        while (k > 0U)
        {
            /* copy second buffer in reversal manner for remaining samples */
            x4 = (q15_t)*px--;
            *pScr2++ = x4;

            /* Decrement loop counter */
            k--;
        }
#endif /*defined (RISCV_MATH_VECTOR)*/
        /* Initialze temporary scratch pointer */
        pScr1 = pScratch1;

        /* Fill (srcBLen - 1U) zeros in scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update temporary scratch pointer */
        pScr1 += (srcBLen - 1U);
#if defined(RISCV_MATH_VECTOR)
        vblkCnt = srcALen; /* Loop counter */
        for (; (l = __riscv_vsetvl_e8m4(vblkCnt)) > 0; vblkCnt -= l)
        {
            vx = __riscv_vwadd_vx_i16m8(__riscv_vle8_v_i8m4(pIn1, l), 0, l);
            pIn1 += l;
            __riscv_vse16_v_i16m8(pScr1, vx, l);
            pScr1 += l;
        }
#else
        /* Copy (srcALen) samples in scratch buffer */
        /* Apply loop unrolling and do 4 Copies simultaneously. */
        k = srcALen >> 2U;

        /* First part of the processing with loop unrolling copies 4 data points at a time.
         ** a second loop below copies for the remaining 1 to 3 samples. */
        while (k > 0U)
        {
            /* copy second buffer in reversal manner */
            x4 = (q15_t)*pIn1++;
            *pScr1++ = x4;
            x4 = (q15_t)*pIn1++;
            *pScr1++ = x4;
            x4 = (q15_t)*pIn1++;
            *pScr1++ = x4;
            x4 = (q15_t)*pIn1++;
            *pScr1++ = x4;
            /* Decrement loop counter */
            k--;
        }

        /* If the count is not a multiple of 4, copy remaining samples here.
         ** No loop unrolling is used. */
        k = srcALen & 0x3U;

        while (k > 0U)
        {
            /* copy second buffer in reversal manner for remaining samples */
            x4 = (q15_t)*pIn1++;
            *pScr1++ = x4;

            /* Decrement the loop counter */
            k--;
        }
#endif /*defined (RISCV_MATH_VECTOR)*/
        /* Fill (srcBLen - 1U) zeros at end of scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update pointer */
        pScr1 += (srcBLen - 1U);

        /* Temporary pointer for scratch2 */
        py = pScratch2;

        /* Initialization of pIn2 pointer */
        pIn2 = (q7_t *)py;

        pScr2 = py;

        pOut = pDst + firstIndex;

        pScratch1 += firstIndex;
#if defined(RISCV_MATH_VECTOR)
        blkCnt = numPoints;
        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;

            uint32_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vint16m4_t vx, vy;
            vint32m1_t temp00m1;
            l = __riscv_vsetvl_e32m1(vblkCnt);
            temp00m1 = __riscv_vmv_v_x_i32m1(0, l);
            for (; (l = __riscv_vsetvl_e16m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle16_v_i16m4(pScr1, l);
                pScr1 += l;
                vy = __riscv_vle16_v_i16m4(pScr2, l);
                pScr2 += l;
                temp00m1 = __riscv_vredsum_vs_i32m8_i32m1(__riscv_vwmul_vv_i32m8(vx, vy, l), temp00m1, l);
            }
            acc0 += __riscv_vmv_x_s_i32m1_i32(temp00m1);

            blkCnt--;

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(acc0 >> 7U, 8));

            /* Initialization of inputB pointer */
            pScr2 = py;

            pScratch1 += 1U;
        }
#else

        /* Actual convolution process starts here */
        blkCnt = (numPoints) >> 2;

        while (blkCnt > 0)
        {
            /* Initialize temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumulators */
            acc0 = 0;
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;

            /* Read two samples from scratch1 buffer */
            x1 = read_q15x2_ia(&pScr1);

            /* Read next two samples from scratch1 buffer */
            x2 = read_q15x2_ia(&pScr1);

            tapCnt = (srcBLen) >> 2U;

            while (tapCnt > 0U)
            {
                /* Read four samples from smaller buffer */
                y1 = read_q15x2_ia(&pScr2);

                /* multiply and accumulate */
                acc0 = __SMLAD(x1, y1, acc0);
                acc2 = __SMLAD(x2, y1, acc2);

                /* pack input data */
                x3 = __PKHBT(x2, x1, 0);

                /* multiply and accumulate */
                acc1 = __SMLADX(x3, y1, acc1);

                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia(&pScr1);

                /* pack input data */
                x3 = __PKHBT(x1, x2, 0);

                acc3 = __SMLADX(x3, y1, acc3);

                /* Read four samples from smaller buffer */
                y1 = read_q15x2_ia(&pScr2);

                acc0 = __SMLAD(x2, y1, acc0);

                acc2 = __SMLAD(x1, y1, acc2);

                acc1 = __SMLADX(x3, y1, acc1);

                x2 = read_q15x2_ia(&pScr1);

                x3 = __PKHBT(x2, x1, 0);

                acc3 = __SMLADX(x3, y1, acc3);

                /* Decrement loop counter */
                tapCnt--;
            }

            /* Update scratch pointer for remaining samples of smaller length sequence */
            pScr1 -= 4U;

            /* apply same above for remaining samples of smaller length sequence */
            tapCnt = (srcBLen) & 3U;

            while (tapCnt > 0U)
            {
                /* accumulate the results */
                acc0 += (*pScr1++ * *pScr2);
                acc1 += (*pScr1++ * *pScr2);
                acc2 += (*pScr1++ * *pScr2);
                acc3 += (*pScr1++ * *pScr2++);

                pScr1 -= 3U;

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* Store the result in the accumulator in the destination buffer. */
            out0 = (q7_t)(__SSAT(acc0 >> 7U, 8));
            out1 = (q7_t)(__SSAT(acc1 >> 7U, 8));
            out2 = (q7_t)(__SSAT(acc2 >> 7U, 8));
            out3 = (q7_t)(__SSAT(acc3 >> 7U, 8));

            write_q7x4_ia(&pOut, __PACKq7(out0, out1, out2, out3));

            /* Initialization of inputB pointer */
            pScr2 = py;

            pScratch1 += 4U;
        }

        blkCnt = (numPoints) & 0x3;

        /* Calculate convolution for remaining samples of Bigger length sequence */
        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;

            tapCnt = (srcBLen) >> 1U;

            while (tapCnt > 0U)
            {

                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia(&pScr1);

                /* Read two samples from smaller buffer */
                y1 = read_q15x2_ia(&pScr2);

                acc0 = __SMLAD(x1, y1, acc0);

                /* Decrement the loop counter */
                tapCnt--;
            }

            tapCnt = (srcBLen) & 1U;

            /* apply same above for remaining samples of smaller length sequence */
            while (tapCnt > 0U)
            {

                /* accumulate the results */
                acc0 += (*pScr1++ * *pScr2++);

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = (q7_t)(__SSAT(acc0 >> 7U, 8));

            /* Initialization of inputB pointer */
            pScr2 = py;

            pScratch1 += 1U;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    return (status);
}

RISCV_DSP_ATTRIBUTE void riscv_copy_q15(
    const q15_t *pSrc,
    q15_t *pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt; /* Loop counter */

#if defined(RISCV_MATH_VECTOR)
    blkCnt = blockSize; /* Loop counter */
    size_t l;
    vint16m8_t v_copy;

    for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l)
    {
        v_copy = __riscv_vle16_v_i16m8(pSrc, l);
        pSrc += l;
        __riscv_vse16_v_i16m8(pDst, v_copy, l);
        pDst += l;
    }
#else
#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 outputs at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* C = A */
#if __RISCV_XLEN == 64
        /* read 4 samples at a time */
        write_q15x4_ia(&pDst, read_q15x4_ia((q15_t **)&pSrc));
#else
        /* read 2 times 2 samples at a time */
        write_q15x2_ia(&pDst, read_q15x2_ia((q15_t **)&pSrc));
        write_q15x2_ia(&pDst, read_q15x2_ia((q15_t **)&pSrc));
#endif /* __RISCV_XLEN == 64 */

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* C = A */

        /* Copy and store result in destination buffer */
        *pDst++ = *pSrc++;

        /* Decrement loop counter */
        blkCnt--;
    }
#endif /* defined(RISCV_MATH_VECTOR) */
}

__STATIC_FORCEINLINE uint64_t __SMLALD(
    uint32_t x,
    uint32_t y,
    uint64_t sum)
{
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x) >> 16) * (((q31_t)y) >> 16)) +
                       (((q63_t)sum))));
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_opt_q15(
    const q15_t *pSrcA,
    uint32_t srcALen,
    const q15_t *pSrcB,
    uint32_t srcBLen,
    q15_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints,
    q15_t *pScratch1,
    q15_t *pScratch2)
{

    q15_t *pOut = pDst;       /* Output pointer */
    q15_t *pScr1 = pScratch1; /* Temporary pointer for scratch1 */
    q15_t *pScr2 = pScratch2; /* Temporary pointer for scratch1 */
    q63_t acc0;               /* Accumulator */
    q31_t x1;                 /* Temporary variables to hold state and coefficient values */
    q31_t y1;                 /* State variables */
    const q15_t *pIn1;        /* InputA pointer */
    const q15_t *pIn2;        /* InputB pointer */
    const q15_t *px;          /* Intermediate inputA pointer */
    q15_t *py;                /* Intermediate inputB pointer */
    uint32_t j, k, blkCnt;    /* Loop counter */
    uint32_t tapCnt;          /* Loop count */
    riscv_status status;      /* Status variable */

#if defined(RISCV_MATH_LOOPUNROLL)
    q63_t acc1, acc2, acc3; /* Accumulator */
    q31_t x2, x3;           /* Temporary variables to hold state and coefficient values */
    q31_t y2;               /* State variables */
#endif

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Temporary pointer for scratch2 */
        py = pScratch2;

        /* pointer to take end of scratch2 buffer */
        pScr2 = pScratch2 + srcBLen - 1;

        /* points to smaller length sequence */
        px = pIn2;

#if defined(RISCV_MATH_VECTOR)
        uint32_t vblkCnt = srcBLen; /* Loop counter */
        size_t l;
        vint16m8_t vx;
        ptrdiff_t bstride = -2;
        for (; (l = __riscv_vsetvl_e16m8(vblkCnt)) > 0; vblkCnt -= l)
        {
            vx = __riscv_vle16_v_i16m8(px, l);
            px += l;
            __riscv_vsse16_v_i16m8(pScr2, bstride, vx, l);
            pScr2 -= l;
        }
#else
#if defined(RISCV_MATH_LOOPUNROLL)

        /* Loop unrolling: Compute 4 outputs at a time */
        k = srcBLen >> 2U;

        /* Copy smaller length input sequence in reverse order into second scratch buffer */
        while (k > 0U)
        {
            /* copy second buffer in reversal manner */
            *pScr2-- = *px++;
            *pScr2-- = *px++;
            *pScr2-- = *px++;
            *pScr2-- = *px++;

            /* Decrement loop counter */
            k--;
        }

        /* Loop unrolling: Compute remaining outputs */
        k = srcBLen & 0x3U;

#else

        /* Initialize k with number of samples */
        k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

        while (k > 0U)
        {
            /* copy second buffer in reversal manner for remaining samples */
            *pScr2-- = *px++;

            /* Decrement loop counter */
            k--;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* Initialze temporary scratch pointer */
        pScr1 = pScratch1;

        /* Assuming scratch1 buffer is aligned by 32-bit */
        /* Fill (srcBLen - 1U) zeros in scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update temporary scratch pointer */
        pScr1 += (srcBLen - 1U);

        /* Copy bigger length sequence(srcALen) samples in scratch1 buffer */

        /* Copy (srcALen) samples in scratch buffer */
        riscv_copy_q15(pIn1, pScr1, srcALen);

        /* Update pointers */
        pScr1 += srcALen;

        /* Fill (srcBLen - 1U) zeros at end of scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update pointer */
        pScr1 += (srcBLen - 1U);

        /* Initialization of pIn2 pointer */
        pIn2 = py;

        pScratch1 += firstIndex;

        pOut = pDst + firstIndex;

        /* Actual convolution process starts here */

#if defined(RISCV_MATH_VECTOR)
        blkCnt = numPoints;
        while (blkCnt > 0)
        {
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;

            uint32_t vblkCnt = srcBLen; /* Loop counter */
            size_t l;
            vint16m4_t vx, vy;
            vint32m1_t temp00m1;
            l = __riscv_vsetvl_e32m1(1);
            temp00m1 = __riscv_vmv_v_x_i32m1(0, l);
            for (; (l = __riscv_vsetvl_e16m4(vblkCnt)) > 0; vblkCnt -= l)
            {
                vx = __riscv_vle16_v_i16m4(pScr1, l);
                pScr1 += l;
                vy = __riscv_vle16_v_i16m4(pIn2, l);
                pIn2 += l;
                temp00m1 = __riscv_vredsum_vs_i32m8_i32m1(__riscv_vwmul_vv_i32m8(vx, vy, l), temp00m1, l);
            }
            acc0 += __riscv_vmv_x_s_i32m1_i32(temp00m1);

            blkCnt--;

            /* The result is in 2.30 format.  Convert to 1.15 with saturation.
             ** Then store the output in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((acc0 >> 15), 16));

            /* Initialization of inputB pointer */
            pIn2 = py;

            pScratch1 += 1U;
        }
#else
#if defined(RISCV_MATH_LOOPUNROLL)

        /* Loop unrolling: Compute 4 outputs at a time */
        blkCnt = (numPoints) >> 2;

        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;

            /* Read two samples from scratch1 buffer */
            x1 = read_q15x2_ia((q15_t **)&pScr1);

            /* Read next two samples from scratch1 buffer */
            x2 = read_q15x2_ia((q15_t **)&pScr1);

            tapCnt = (srcBLen) >> 2U;

            while (tapCnt > 0U)
            {

                /* Read four samples from smaller buffer */
                y1 = read_q15x2_ia((q15_t **)&pIn2);
                y2 = read_q15x2_ia((q15_t **)&pIn2);

                /* multiply and accumulate */
                acc0 = __SMLALD(x1, y1, acc0);
                acc2 = __SMLALD(x2, y1, acc2);

                /* pack input data */
                x3 = __PKHBT(x2, x1, 0);

                /* multiply and accumulate */
                acc1 = __SMLALDX(x3, y1, acc1);

                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia((q15_t **)&pScr1);

                /* multiply and accumulate */
                acc0 = __SMLALD(x2, y2, acc0);
                acc2 = __SMLALD(x1, y2, acc2);

                /* pack input data */
                x3 = __PKHBT(x1, x2, 0);

                acc3 = __SMLALDX(x3, y1, acc3);
                acc1 = __SMLALDX(x3, y2, acc1);

                x2 = read_q15x2_ia((q15_t **)&pScr1);

                x3 = __PKHBT(x2, x1, 0);

                acc3 = __SMLALDX(x3, y2, acc3);

                /* Decrement loop counter */
                tapCnt--;
            }

            /* Update scratch pointer for remaining samples of smaller length sequence */
            pScr1 -= 4U;

            /* apply same above for remaining samples of smaller length sequence */
            tapCnt = (srcBLen) & 3U;

            while (tapCnt > 0U)
            {
                /* accumulate the results */
                acc0 += (*pScr1++ * *pIn2);
                acc1 += (*pScr1++ * *pIn2);
                acc2 += (*pScr1++ * *pIn2);
                acc3 += (*pScr1++ * *pIn2++);

                pScr1 -= 3U;

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* Store the results in the accumulators in the destination buffer. */
            write_q15x2_ia(&pOut, __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16));
            write_q15x2_ia(&pOut, __PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16));

            /* Initialization of inputB pointer */
            pIn2 = py;

            pScratch1 += 4U;
        }

        /* Loop unrolling: Compute remaining outputs */
        blkCnt = numPoints & 0x3;

#else

        /* Initialize blkCnt with number of samples */
        blkCnt = numPoints;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

        /* Calculate convolution for remaining samples of Bigger length sequence */
        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;

            tapCnt = (srcBLen) >> 1U;

            while (tapCnt > 0U)
            {
                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia((q15_t **)&pScr1);

                /* Read two samples from smaller buffer */
                y1 = read_q15x2_ia((q15_t **)&pIn2);

                acc0 = __SMLALD(x1, y1, acc0);

                /* Decrement the loop counter */
                tapCnt--;
            }

            tapCnt = (srcBLen) & 1U;

            /* apply same above for remaining samples of smaller length sequence */
            while (tapCnt > 0U)
            {
                /* accumulate the results */
                acc0 += (*pScr1++ * *pIn2++);

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* The result is in 2.30 format.  Convert to 1.15 with saturation.
             ** Then store the output in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((acc0 >> 15), 16));

            /* Initialization of inputB pointer */
            pIn2 = py;

            pScratch1 += 1U;
        }
#endif /* defined (RISCV_MATH_VECTOR) */
        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);
}

#define ARRAYA_SIZE_Q15 1024
#define ARRAYB_SIZE_Q15 1024

static q15_t q15_pScratch1[max(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15) + 2 * min(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15) - 2] = {};
static q15_t q15_pScratch2[min(ARRAYA_SIZE_Q15, ARRAYB_SIZE_Q15)] = {};

RISCV_DSP_ATTRIBUTE riscv_status riscv_conv_partial_fast_opt_q15(
    const q15_t *pSrcA,
    uint32_t srcALen,
    const q15_t *pSrcB,
    uint32_t srcBLen,
    q15_t *pDst,
    uint32_t firstIndex,
    uint32_t numPoints,
    q15_t *pScratch1,
    q15_t *pScratch2)
{
#if defined(RISCV_MATH_VECTOR)
    return riscv_conv_partial_opt_q15(pSrcA, srcALen, pSrcB, srcBLen, pDst, firstIndex, numPoints, pScratch1, pScratch2);
#else
    q15_t *pOut = pDst;       /* Output pointer */
    q15_t *pScr1 = pScratch1; /* Temporary pointer for scratch1 */
    q15_t *pScr2 = pScratch2; /* Temporary pointer for scratch1 */
    q31_t acc0;               /* Accumulator */
    const q15_t *pIn1;        /* InputA pointer */
    const q15_t *pIn2;        /* InputB pointer */
    const q15_t *px;          /* Intermediate inputA pointer */
    q15_t *py;                /* Intermediate inputB pointer */
    uint32_t j, k, blkCnt;    /* Loop counter */
    uint32_t tapCnt;          /* Loop count */
    riscv_status status;      /* Status variable */
    q31_t x1;                 /* Temporary variables to hold state and coefficient values */
    q31_t y1;                 /* State variables */

#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t acc1, acc2, acc3; /* Accumulator */
    q31_t x2, x3;           /* Temporary variables to hold state and coefficient values */
    q31_t y2;               /* State variables */
#endif

    /* Check for range of output samples to be calculated */
    if ((firstIndex + numPoints) > ((srcALen + (srcBLen - 1U))))
    {
        /* Set status as RISCV_MATH_ARGUMENT_ERROR */
        status = RISCV_MATH_ARGUMENT_ERROR;
    }
    else
    {
        /* The algorithm implementation is based on the lengths of the inputs. */
        /* srcB is always made to slide across srcA. */
        /* So srcBLen is always considered as shorter or equal to srcALen */
        if (srcALen >= srcBLen)
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcA;

            /* Initialization of inputB pointer */
            pIn2 = pSrcB;
        }
        else
        {
            /* Initialization of inputA pointer */
            pIn1 = pSrcB;

            /* Initialization of inputB pointer */
            pIn2 = pSrcA;

            /* srcBLen is always considered as shorter or equal to srcALen */
            j = srcBLen;
            srcBLen = srcALen;
            srcALen = j;
        }

        /* Temporary pointer for scratch2 */
        py = pScratch2;

        /* pointer to take end of scratch2 buffer */
        pScr2 = pScratch2 + srcBLen - 1;

        /* points to smaller length sequence */
        px = pIn2;

#if defined(RISCV_MATH_LOOPUNROLL)

        /* Loop unrolling: Compute 4 outputs at a time */
        k = srcBLen >> 2U;

        /* Copy smaller length input sequence in reverse order into second scratch buffer */
        while (k > 0U)
        {
            /* copy second buffer in reversal manner */
            *pScr2-- = *px++;
            *pScr2-- = *px++;
            *pScr2-- = *px++;
            *pScr2-- = *px++;

            /* Decrement loop counter */
            k--;
        }

        /* Loop unrolling: Compute remaining outputs */
        k = srcBLen & 0x3U;

#else

        /* Initialize k with number of samples */
        k = srcBLen;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

        while (k > 0U)
        {
            /* copy second buffer in reversal manner for remaining samples */
            *pScr2-- = *px++;

            /* Decrement loop counter */
            k--;
        }

        /* Initialze temporary scratch pointer */
        pScr1 = pScratch1;

        /* Assuming scratch1 buffer is aligned by 32-bit */
        /* Fill (srcBLen - 1U) zeros in scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update temporary scratch pointer */
        pScr1 += (srcBLen - 1U);

        /* Copy bigger length sequence(srcALen) samples in scratch1 buffer */

        /* Copy (srcALen) samples in scratch buffer */
        riscv_copy_q15(pIn1, pScr1, srcALen);

        /* Update pointers */
        pScr1 += srcALen;

        /* Fill (srcBLen - 1U) zeros at end of scratch buffer */
        riscv_fill_q15(0, pScr1, (srcBLen - 1U));

        /* Update pointer */
        pScr1 += (srcBLen - 1U);

        /* Initialization of pIn2 pointer */
        pIn2 = py;

        pScratch1 += firstIndex;

        pOut = pDst + firstIndex;

        /* Actual convolution process starts here */

#if defined(RISCV_MATH_LOOPUNROLL)

        /* Loop unrolling: Compute 4 outputs at a time */
        blkCnt = (numPoints) >> 2;

        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;

            /* Read two samples from scratch1 buffer */
            x1 = read_q15x2_ia(&pScr1);

            /* Read next two samples from scratch1 buffer */
            x2 = read_q15x2_ia(&pScr1);

            tapCnt = (srcBLen) >> 2U;

            while (tapCnt > 0U)
            {

                /* Read four samples from smaller buffer */
                y1 = read_q15x2_ia((q15_t **)&pIn2);
                y2 = read_q15x2_ia((q15_t **)&pIn2);

                /* multiply and accumulate */
                acc0 = __SMLAD(x1, y1, acc0);
                acc2 = __SMLAD(x2, y1, acc2);

                /* pack input data */
                x3 = __PKHBT(x2, x1, 0);

                /* multiply and accumulate */
                acc1 = __SMLADX(x3, y1, acc1);

                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia(&pScr1);

                /* multiply and accumulate */
                acc0 = __SMLAD(x2, y2, acc0);
                acc2 = __SMLAD(x1, y2, acc2);

                /* pack input data */
                x3 = __PKHBT(x1, x2, 0);

                acc3 = __SMLADX(x3, y1, acc3);
                acc1 = __SMLADX(x3, y2, acc1);

                x2 = read_q15x2_ia(&pScr1);

                x3 = __PKHBT(x2, x1, 0);

                /* multiply and accumulate */
                acc3 = __SMLADX(x3, y2, acc3);

                /* Decrement loop counter */
                tapCnt--;
            }

            /* Update scratch pointer for remaining samples of smaller length sequence */
            pScr1 -= 4U;

            /* apply same above for remaining samples of smaller length sequence */
            tapCnt = (srcBLen) & 3U;

            while (tapCnt > 0U)
            {
                /* accumulate the results */
                acc0 += (*pScr1++ * *pIn2);
                acc1 += (*pScr1++ * *pIn2);
                acc2 += (*pScr1++ * *pIn2);
                acc3 += (*pScr1++ * *pIn2++);

                pScr1 -= 3U;

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* Store the results in the accumulators in the destination buffer. */
            write_q15x2_ia(&pOut, __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16));
            write_q15x2_ia(&pOut, __PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16));

            /* Initialization of inputB pointer */
            pIn2 = py;

            pScratch1 += 4U;
        }

        /* Loop unrolling: Compute remaining outputs */
        blkCnt = numPoints & 0x3;

#else

        /* Initialize blkCnt with number of samples */
        blkCnt = numPoints;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

        /* Calculate convolution for remaining samples of Bigger length sequence */
        while (blkCnt > 0)
        {
            /* Initialze temporary scratch pointer as scratch1 */
            pScr1 = pScratch1;

            /* Clear Accumlators */
            acc0 = 0;

            tapCnt = (srcBLen) >> 1U;

            while (tapCnt > 0U)
            {
                /* Read next two samples from scratch1 buffer */
                x1 = read_q15x2_ia(&pScr1);

                /* Read two samples from smaller buffer */
                y1 = read_q15x2_ia((q15_t **)&pIn2);

                /* multiply and accumulate */
                acc0 = __SMLAD(x1, y1, acc0);

                /* Decrement loop counter */
                tapCnt--;
            }

            tapCnt = (srcBLen) & 1U;

            /* apply same above for remaining samples of smaller length sequence */
            while (tapCnt > 0U)
            {
                /* accumulate the results */
                acc0 += (*pScr1++ * *pIn2++);

                /* Decrement loop counter */
                tapCnt--;
            }

            blkCnt--;

            /* The result is in 2.30 format.  Convert to 1.15 with saturation.
             ** Then store the output in the destination buffer. */
            *pOut++ = (q15_t)(__SSAT((acc0 >> 15), 16));

            /* Initialization of inputB pointer */
            pIn2 = py;

            pScratch1 += 1U;
        }

        /* Set status as RISCV_MATH_SUCCESS */
        status = RISCV_MATH_SUCCESS;
    }

    /* Return to application */
    return (status);
#endif /* defined (RISCV_MATH_VECTOR) */
}

#define TEST_LENGTH_SAMPLES 1024
#define NUM_TAPS 32 /* Must be even */

static float32_t testInput_f32_50Hz_200Hz[TEST_LENGTH_SAMPLES] = {};

static float32_t firCoeffs32LP[NUM_TAPS] = {
    -0.001822523074f, -0.001587929321f, 1.226008847e-18f, 0.003697750857f, 0.008075430058f,
    0.008530221879f, -4.273456581e-18f, -0.01739769801f, -0.03414586186f, -0.03335915506f,
    8.073562366e-18f, 0.06763084233f, 0.1522061825f, 0.2229246944f, 0.2504960895f,
    0.2229246944f, 0.1522061825f, 0.06763084233f, 8.073562366e-18f, -0.03335915506f,
    -0.03414586186f, -0.01739769801f, -4.273456581e-18f, 0.008530221879f, 0.008075430058f,
    0.003697750857f, 1.226008847e-18f, -0.001587929321f, -0.001822523074f, 0.0f,
    0.12343084233f, -0.0345061825f};

/**
 * @brief 8-bit fractional data type in 1.7 format.
 */
typedef int8_t q7_t;

/**
 * @brief 16-bit fractional data type in 1.15 format.
 */
typedef int16_t q15_t;

/**
 * @brief 64-bit floating-point type definition.
 */
typedef double float64_t;

typedef struct
{
    uint16_t numTaps;    /**< number of filter coefficients in the filter. */
    q7_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q7_t *pCoeffs; /**< points to the coefficient array. The array is of length numTaps.*/
} riscv_fir_instance_q7;

/**
 * @brief Instance structure for the Q15 FIR filter.
 */
typedef struct
{
    uint16_t numTaps;     /**< number of filter coefficients in the filter. */
    q15_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q15_t *pCoeffs; /**< points to the coefficient array. The array is of length numTaps.*/
} riscv_fir_instance_q15;

/**
 * @brief Instance structure for the Q31 FIR filter.
 */
typedef struct
{
    uint16_t numTaps;     /**< number of filter coefficients in the filter. */
    q31_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const q31_t *pCoeffs; /**< points to the coefficient array. The array is of length numTaps. */
} riscv_fir_instance_q31;

/**
 * @brief Instance structure for the floating-point FIR filter.
 */
typedef struct
{
    uint16_t numTaps;         /**< number of filter coefficients in the filter. */
    float32_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const float32_t *pCoeffs; /**< points to the coefficient array. The array is of length numTaps. */
} riscv_fir_instance_f32;

/**
 * @brief Instance structure for the floating-point FIR filter.
 */
typedef struct
{
    uint16_t numTaps;         /**< number of filter coefficients in the filter. */
    float64_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    const float64_t *pCoeffs; /**< points to the coefficient array. The array is of length numTaps. */
} riscv_fir_instance_f64;

RISCV_DSP_ATTRIBUTE void riscv_fir_init_f32(
    riscv_fir_instance_f32 *S,
    uint16_t numTaps,
    const float32_t *pCoeffs,
    float32_t *pState,
    uint32_t blockSize)
{
    /* Assign filter taps */
    S->numTaps = numTaps;

    /* Assign coefficient pointer */
    S->pCoeffs = pCoeffs;

    /* Clear state buffer. The size is always (blockSize + numTaps - 1) */
    memset(pState, 0, (numTaps + (blockSize - 1U)) * sizeof(float32_t));
    /* Assign state pointer */
    S->pState = pState;
}

RISCV_DSP_ATTRIBUTE void riscv_fir_f32(
    const riscv_fir_instance_f32 *S,
    const float32_t *pSrc,
    float32_t *pDst,
    uint32_t blockSize)
{
    float32_t *pState = S->pState;         /* State pointer */
    const float32_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    float32_t *pStateCurnt;                /* Points to the current sample of the state */
    float32_t *px;                         /* Temporary pointer for state buffer */
    const float32_t *pb;                   /* Temporary pointer for coefficient buffer */
    float32_t acc0;                        /* Accumulator */
    uint32_t numTaps = S->numTaps;         /* Number of filter coefficients in the filter */
    uint32_t i, tapCnt, blkCnt;            /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    float32_t acc1, acc2, acc3, acc4, acc5, acc6, acc7; /* Accumulators */
    float32_t x0, x1, x2, x3, x4, x5, x6, x7;           /* Temporary variables to hold state values */
    float32_t c0;                                       /* Temporary variable to hold coefficient value */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_VECTOR)
    uint32_t j;
    size_t l;
    vfloat32m8_t vx, vres0m8;
    float32_t *pOut = pDst;
    /* Copy samples into state buffer */
    riscv_copy_f32(pSrc, pStateCurnt, blockSize);
    for (i = blockSize; i > 0; i -= l)
    {
        l = __riscv_vsetvl_e32m8(i);
        vx = __riscv_vle32_v_f32m8(pState, l);
        pState += l;
        vres0m8 = __riscv_vfmv_v_f_f32m8(0.0, l);
        for (j = 0; j < numTaps; j++)
        {
            vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pCoeffs + j), vx, l);
            vx = __riscv_vfslide1down_vf_f32m8(vx, *(pState + j), l);
        }
        __riscv_vse32_v_f32m8(pOut, vres0m8, l);
        pOut += l;
    }
    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;
    /* Copy data */
    riscv_copy_f32(pState, pStateCurnt, numTaps - 1);
#else
#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 8 output values simultaneously.
     * The variables acc0 ... acc7 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */

    blkCnt = blockSize >> 3U;

    while (blkCnt > 0U)
    {
        /* Copy 4 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0.0f;
        acc1 = 0.0f;
        acc2 = 0.0f;
        acc3 = 0.0f;
        acc4 = 0.0f;
        acc5 = 0.0f;
        acc6 = 0.0f;
        acc7 = 0.0f;

        /* Initialize state pointer */
        px = pState;

        /* Initialize coefficient pointer */
        pb = pCoeffs;

        /* This is separated from the others to avoid
         * a call to __aeabi_memmove which would be slower
         */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Read the first 7 samples from the state buffer:  x[n-numTaps], x[n-numTaps-1], x[n-numTaps-2] */
        x0 = *px++;
        x1 = *px++;
        x2 = *px++;
        x3 = *px++;
        x4 = *px++;
        x5 = *px++;
        x6 = *px++;

        /* Loop unrolling: process 8 taps at a time. */
        tapCnt = numTaps >> 3U;

        while (tapCnt > 0U)
        {
            /* Read the b[numTaps-1] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-3] sample */
            x7 = *(px++);

            /* acc0 +=  b[numTaps-1] * x[n-numTaps] */
            acc0 += x0 * c0;

            /* acc1 +=  b[numTaps-1] * x[n-numTaps-1] */
            acc1 += x1 * c0;

            /* acc2 +=  b[numTaps-1] * x[n-numTaps-2] */
            acc2 += x2 * c0;

            /* acc3 +=  b[numTaps-1] * x[n-numTaps-3] */
            acc3 += x3 * c0;

            /* acc4 +=  b[numTaps-1] * x[n-numTaps-4] */
            acc4 += x4 * c0;

            /* acc1 +=  b[numTaps-1] * x[n-numTaps-5] */
            acc5 += x5 * c0;

            /* acc2 +=  b[numTaps-1] * x[n-numTaps-6] */
            acc6 += x6 * c0;

            /* acc3 +=  b[numTaps-1] * x[n-numTaps-7] */
            acc7 += x7 * c0;

            /* Read the b[numTaps-2] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-4] sample */
            x0 = *(px++);

            /* Perform the multiply-accumulate */
            acc0 += x1 * c0;
            acc1 += x2 * c0;
            acc2 += x3 * c0;
            acc3 += x4 * c0;
            acc4 += x5 * c0;
            acc5 += x6 * c0;
            acc6 += x7 * c0;
            acc7 += x0 * c0;

            /* Read the b[numTaps-3] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-5] sample */
            x1 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x2 * c0;
            acc1 += x3 * c0;
            acc2 += x4 * c0;
            acc3 += x5 * c0;
            acc4 += x6 * c0;
            acc5 += x7 * c0;
            acc6 += x0 * c0;
            acc7 += x1 * c0;

            /* Read the b[numTaps-4] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-6] sample */
            x2 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x3 * c0;
            acc1 += x4 * c0;
            acc2 += x5 * c0;
            acc3 += x6 * c0;
            acc4 += x7 * c0;
            acc5 += x0 * c0;
            acc6 += x1 * c0;
            acc7 += x2 * c0;

            /* Read the b[numTaps-4] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-6] sample */
            x3 = *(px++);
            /* Perform the multiply-accumulates */
            acc0 += x4 * c0;
            acc1 += x5 * c0;
            acc2 += x6 * c0;
            acc3 += x7 * c0;
            acc4 += x0 * c0;
            acc5 += x1 * c0;
            acc6 += x2 * c0;
            acc7 += x3 * c0;

            /* Read the b[numTaps-4] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-6] sample */
            x4 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x5 * c0;
            acc1 += x6 * c0;
            acc2 += x7 * c0;
            acc3 += x0 * c0;
            acc4 += x1 * c0;
            acc5 += x2 * c0;
            acc6 += x3 * c0;
            acc7 += x4 * c0;

            /* Read the b[numTaps-4] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-6] sample */
            x5 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x6 * c0;
            acc1 += x7 * c0;
            acc2 += x0 * c0;
            acc3 += x1 * c0;
            acc4 += x2 * c0;
            acc5 += x3 * c0;
            acc6 += x4 * c0;
            acc7 += x5 * c0;

            /* Read the b[numTaps-4] coefficient */
            c0 = *(pb++);

            /* Read x[n-numTaps-6] sample */
            x6 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x7 * c0;
            acc1 += x0 * c0;
            acc2 += x1 * c0;
            acc3 += x2 * c0;
            acc4 += x3 * c0;
            acc5 += x4 * c0;
            acc6 += x5 * c0;
            acc7 += x6 * c0;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* Loop unrolling: Compute remaining outputs */
        tapCnt = numTaps % 0x8U;

        while (tapCnt > 0U)
        {
            /* Read coefficients */
            c0 = *(pb++);

            /* Fetch 1 state variable */
            x7 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += x0 * c0;
            acc1 += x1 * c0;
            acc2 += x2 * c0;
            acc3 += x3 * c0;
            acc4 += x4 * c0;
            acc5 += x5 * c0;
            acc6 += x6 * c0;
            acc7 += x7 * c0;

            /* Reuse the present sample states for next sample */
            x0 = x1;
            x1 = x2;
            x2 = x3;
            x3 = x4;
            x4 = x5;
            x5 = x6;
            x6 = x7;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* Advance the state pointer by 8 to process the next group of 8 samples */
        pState = pState + 8;

        /* The results in the 8 accumulators, store in the destination buffer. */
        *pDst++ = acc0;
        *pDst++ = acc1;
        *pDst++ = acc2;
        *pDst++ = acc3;
        *pDst++ = acc4;
        *pDst++ = acc5;
        *pDst++ = acc6;
        *pDst++ = acc7;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize & 0x7U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy one sample at a time into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0.0f;

        /* Initialize state pointer */
        px = pState;

        /* Initialize Coefficient pointer */
        pb = pCoeffs;

        i = numTaps;
        /* Perform the multiply-accumulates */
        while (i > 0U)
        {
            /* acc =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0] */
            acc0 += *px++ * *pb++;

            i--;
        }

        /* Store result in destination buffer. */
        *pDst++ = acc0;

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */
    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }
#endif /* defined (RISCV_MATH_VECTOR) */
}

#define TEST_LENGTH_SAMPLES_F64 1024
#define NUM_TAPS_F64 32 /* Must be even */

static float64_t testInput_f64_50Hz_200Hz[TEST_LENGTH_SAMPLES_F64] = {};

static float64_t firCoeffs64LP[NUM_TAPS_F64] = {};

static void generate_rand_f64(float64_t *src, int length)
{
    do_srand();
    for (int i = 0; i < length; i++)
    {
        src[i] = (float64_t)((rand() % Q31_MAX - Q31_MAX / 2) * 1.0 / Q31_MAX);
    }
}

RISCV_DSP_ATTRIBUTE void riscv_fir_init_f64(
    riscv_fir_instance_f64 *S,
    uint16_t numTaps,
    const float64_t *pCoeffs,
    float64_t *pState,
    uint32_t blockSize)
{
    /* Assign filter taps */
    S->numTaps = numTaps;

    /* Assign coefficient pointer */
    S->pCoeffs = pCoeffs;

    /* Clear state buffer. The size is always (blockSize + numTaps - 1) */
    memset(pState, 0, (numTaps + (blockSize - 1U)) * sizeof(float64_t));
    /* Assign state pointer */
    S->pState = pState;
}

RISCV_DSP_ATTRIBUTE void riscv_fir_f64(
    const riscv_fir_instance_f64 *S,
    const float64_t *pSrc,
    float64_t *pDst,
    uint32_t blockSize)
{
    float64_t *pState = S->pState;         /* State pointer */
    const float64_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    float64_t *pStateCurnt;                /* Points to the current sample of the state */
    float64_t *px;                         /* Temporary pointer for state buffer */
    const float64_t *pb;                   /* Temporary pointer for coefficient buffer */
    float64_t acc0;                        /* Accumulator */
    uint32_t numTaps = S->numTaps;         /* Number of filter coefficients in the filter */
    uint32_t i, tapCnt, blkCnt;            /* Loop counters */

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

    while (blkCnt > 0U)
    {
        /* Copy one sample at a time into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0.;

        /* Initialize state pointer */
        px = pState;

        /* Initialize Coefficient pointer */
        pb = pCoeffs;

        i = numTaps;

        /* Perform the multiply-accumulates */
        while (i > 0U)
        {
            /* acc =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0] */
            acc0 += *px++ * *pb++;

            i--;
        }

        /* Store result in destination buffer. */
        *pDst++ = acc0;

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
     Now copy the last numTaps - 1 samples to the start of the state buffer.
     This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }
}

#define TEST_LENGTH_SAMPLES_Q7 1024
#define NUM_TAPS_Q7 32 /* Must be even */

static q7_t testInput_q7_50Hz_200Hz[TEST_LENGTH_SAMPLES_Q7] = {};

static q7_t firCoeffLP_q7[NUM_TAPS_Q7] = {};

RISCV_DSP_ATTRIBUTE void riscv_float_to_q7(
    const float32_t *pSrc,
    q7_t *pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt;             /* Loop counter */
    const float32_t *pIn = pSrc; /* Source pointer */

#if defined(RISCV_MATH_VECTOR)
    blkCnt = blockSize; /* Loop counter */
    size_t l;
    vfloat32m8_t v_in;
    vint8m2_t v_out;
    for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l)
    {
        v_in = __riscv_vle32_v_f32m8(pIn, l);
        pIn += l;
#ifdef RISCV_MATH_ROUNDING
        v_out = __riscv_vnclip_wx_i8m2(__riscv_vnclip_wx_i16m4(__riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(v_in, 128.0f, l), l), 0U, __RISCV_VXRM_RNU, l), 0U, __RISCV_VXRM_RNU, l);
#else
        v_out = __riscv_vnclip_wx_i8m2(__riscv_vnclip_wx_i16m4(__riscv_vfcvt_rtz_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(v_in, 128.0f, l), l), 0U, __RISCV_VXRM_RNU, l), 0U, __RISCV_VXRM_RNU, l);
#endif
        __riscv_vse8_v_i8m2(pDst, v_out, l);
        pDst += l;
    }
#else

#ifdef RISCV_MATH_ROUNDING
    float32_t in;
#endif /* #ifdef RISCV_MATH_ROUNDING */

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 outputs at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* C = A * 128 */

        /* Convert from float to q7 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 128);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q7_t)(__SSAT((q15_t)(in), 8));

        in = (*pIn++ * 128);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q7_t)(__SSAT((q15_t)(in), 8));

        in = (*pIn++ * 128);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q7_t)(__SSAT((q15_t)(in), 8));

        in = (*pIn++ * 128);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q7_t)(__SSAT((q15_t)(in), 8));

#else

        *pDst++ = __SSAT((q31_t)(*pIn++ * 128.0f), 8);
        *pDst++ = __SSAT((q31_t)(*pIn++ * 128.0f), 8);
        *pDst++ = __SSAT((q31_t)(*pIn++ * 128.0f), 8);
        *pDst++ = __SSAT((q31_t)(*pIn++ * 128.0f), 8);

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* C = A * 128 */

        /* Convert from float to q7 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 128);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q7_t)(__SSAT((q15_t)(in), 8));

#else

        *pDst++ = (q7_t)__SSAT((q31_t)(*pIn++ * 128.0f), 8);

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }
#endif /* defined(RISCV_MATH_VECTOR) */
}

RISCV_DSP_ATTRIBUTE void riscv_fir_init_q7(
    riscv_fir_instance_q7 *S,
    uint16_t numTaps,
    const q7_t *pCoeffs,
    q7_t *pState,
    uint32_t blockSize)
{
    /* Assign filter taps */
    S->numTaps = numTaps;

    /* Assign coefficient pointer */
    S->pCoeffs = pCoeffs;

    /* Clear state buffer. The size is always (blockSize + numTaps - 1) */
    memset(pState, 0, (numTaps + (blockSize - 1U)) * sizeof(q7_t));

    /* Assign state pointer */
    S->pState = pState;
}

RISCV_DSP_ATTRIBUTE void riscv_fir_q7(
    const riscv_fir_instance_q7 *S,
    const q7_t *pSrc,
    q7_t *pDst,
    uint32_t blockSize)
{
    q7_t *pState = S->pState;         /* State pointer */
    const q7_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    q7_t *pStateCurnt;                /* Points to the current sample of the state */
    q7_t *px;                         /* Temporary pointer for state buffer */
    const q7_t *pb;                   /* Temporary pointer for coefficient buffer */
    q31_t acc0;                       /* Accumulators */
    uint32_t numTaps = S->numTaps;    /* Number of filter coefficients in the filter */
    uint32_t i, tapCnt, blkCnt;       /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t acc1, acc2, acc3;  /* Accumulators */
    q7_t x0, x1, x2, x3, c0; /* Temporary variables to hold state */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_VECTOR)
    uint32_t j;
    size_t l;
    vint8m2_t vx;
    vint32m8_t vres0m8;
    q7_t *pOut = pDst;
    /* Copy samples into state buffer */
    riscv_copy_q7(pSrc, pStateCurnt, blockSize);
    for (i = blockSize; i > 0; i -= l)
    {
        l = __riscv_vsetvl_e8m2(i);
        vx = __riscv_vle8_v_i8m2(pState, l);
        pState += l;
        vres0m8 = __riscv_vmv_v_x_i32m8(0, l);
        for (j = 0; j < numTaps; j++)
        {
            vres0m8 = __riscv_vwmacc_vx_i32m8(vres0m8, *(pCoeffs + j), __riscv_vwadd_vx_i16m4(vx, 0, l), l);
            vx = __riscv_vslide1down_vx_i8m2(vx, *(pState + j), l);
        }
        __riscv_vse8_v_i8m2(pOut, __riscv_vnclip_wx_i8m2(__riscv_vnsra_wx_i16m4(vres0m8, 7, l), 0, __RISCV_VXRM_RNU, l), l);
        pOut += l;
    }
    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

    /* Copy data */
    riscv_copy_q7(pState, pStateCurnt, numTaps - 1);
#else

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 output values simultaneously.
     * The variables acc0 ... acc3 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* Copy 4 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize coefficient pointer */
        pb = pCoeffs;

        /* Read the first 3 samples from the state buffer:
         *  x[n-numTaps], x[n-numTaps-1], x[n-numTaps-2] */
        x0 = *px++;
        x1 = *px++;
        x2 = *px++;

        /* Loop unrolling. Process 4 taps at a time. */
        tapCnt = numTaps >> 2U;

        /* Loop over the number of taps.  Unroll by a factor of 4.
           Repeat until we've computed numTaps-4 coefficients. */
        while (tapCnt > 0U)
        {
            /* Read the b[numTaps] coefficient */
            c0 = *pb;

            /* Read x[n-numTaps-3] sample */
            x3 = *px;

            /* acc0 +=  b[numTaps] * x[n-numTaps] */
            acc0 += ((q15_t)x0 * c0);

            /* acc1 +=  b[numTaps] * x[n-numTaps-1] */
            acc1 += ((q15_t)x1 * c0);

            /* acc2 +=  b[numTaps] * x[n-numTaps-2] */
            acc2 += ((q15_t)x2 * c0);

            /* acc3 +=  b[numTaps] * x[n-numTaps-3] */
            acc3 += ((q15_t)x3 * c0);

            /* Read the b[numTaps-1] coefficient */
            c0 = *(pb + 1U);

            /* Read x[n-numTaps-4] sample */
            x0 = *(px + 1U);

            /* Perform the multiply-accumulates */
            acc0 += ((q15_t)x1 * c0);
            acc1 += ((q15_t)x2 * c0);
            acc2 += ((q15_t)x3 * c0);
            acc3 += ((q15_t)x0 * c0);

            /* Read the b[numTaps-2] coefficient */
            c0 = *(pb + 2U);

            /* Read x[n-numTaps-5] sample */
            x1 = *(px + 2U);

            /* Perform the multiply-accumulates */
            acc0 += ((q15_t)x2 * c0);
            acc1 += ((q15_t)x3 * c0);
            acc2 += ((q15_t)x0 * c0);
            acc3 += ((q15_t)x1 * c0);

            /* Read the b[numTaps-3] coefficients */
            c0 = *(pb + 3U);

            /* Read x[n-numTaps-6] sample */
            x2 = *(px + 3U);

            /* Perform the multiply-accumulates */
            acc0 += ((q15_t)x3 * c0);
            acc1 += ((q15_t)x0 * c0);
            acc2 += ((q15_t)x1 * c0);
            acc3 += ((q15_t)x2 * c0);

            /* update coefficient pointer */
            pb += 4U;
            px += 4U;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* If the filter length is not a multiple of 4, compute the remaining filter taps */
        tapCnt = numTaps & 0x3U;

        while (tapCnt > 0U)
        {
            /* Read coefficients */
            c0 = *(pb++);

            /* Fetch 1 state variable */
            x3 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += ((q15_t)x0 * c0);
            acc1 += ((q15_t)x1 * c0);
            acc2 += ((q15_t)x2 * c0);
            acc3 += ((q15_t)x3 * c0);

            /* Reuse the present sample states for next sample */
            x0 = x1;
            x1 = x2;
            x2 = x3;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* The results in the 4 accumulators are in 2.62 format. Convert to 1.31
           Then store the 4 outputs in the destination buffer. */
        acc0 = __SSAT((acc0 >> 7U), 8);
        *pDst++ = acc0;
        acc1 = __SSAT((acc1 >> 7U), 8);
        *pDst++ = acc1;
        acc2 = __SSAT((acc2 >> 7U), 8);
        *pDst++ = acc2;
        acc3 = __SSAT((acc3 >> 7U), 8);
        *pDst++ = acc3;

        /* Advance the state pointer by 4 to process the next group of 4 samples */
        pState = pState + 4U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy one sample at a time into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize Coefficient pointer */
        pb = pCoeffs;

        i = numTaps;
        /* Perform the multiply-accumulates */
        while (i > 0U)
        {
            acc0 += (q15_t) * (px++) * (*(pb++));
            i--;
        }

        /* The result is in 2.14 format. Convert to 1.7
           Then store the output in the destination buffer. */
        *pDst++ = __SSAT((acc0 >> 7U), 8);

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement the loop counter */
        tapCnt--;
    }
#endif /* defined (RISCV_MATH_VECTOR) */
}

#define TEST_LENGTH_SAMPLES_Q15 1024
#define NUM_TAPS_Q15 32 /* Must be even */

static q15_t testInput_q15_50Hz_200Hz[TEST_LENGTH_SAMPLES_Q15] = {};

static q15_t firCoeffLP_q15[NUM_TAPS_Q15] = {};

RISCV_DSP_ATTRIBUTE void riscv_float_to_q15(
    const float32_t *pSrc,
    q15_t *pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt;             /* Loop counter */
    const float32_t *pIn = pSrc; /* Source pointer */

#if defined(RISCV_MATH_VECTOR)
    blkCnt = blockSize; /* Loop counter */
    size_t l;
    vfloat32m8_t v_in;
    vint16m4_t v_out;
    for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l)
    {
        v_in = __riscv_vle32_v_f32m8(pIn, l);
        pIn += l;
#ifdef RISCV_MATH_ROUNDING
        v_out = __riscv_vnclip_wx_i16m4(__riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(v_in, 32768.0f, l), l), 0, __RISCV_VXRM_RNU, l);
#else
        v_out = __riscv_vnclip_wx_i16m4(__riscv_vfcvt_rtz_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(v_in, 32768.0f, l), l), 0, __RISCV_VXRM_RNU, l);
#endif
        __riscv_vse16_v_i16m4(pDst, v_out, l);
        pDst += l;
    }
#else

#ifdef RISCV_MATH_ROUNDING
    float32_t in;
#endif /* #ifdef RISCV_MATH_ROUNDING */

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 outputs at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* C = A * 32768 */

        /* convert from float to Q15 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 32768.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q15_t)(__SSAT((q31_t)(in), 16));

        in = (*pIn++ * 32768.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q15_t)(__SSAT((q31_t)(in), 16));

        in = (*pIn++ * 32768.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q15_t)(__SSAT((q31_t)(in), 16));

        in = (*pIn++ * 32768.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q15_t)(__SSAT((q31_t)(in), 16));

#else

        *pDst++ = (q15_t)__SSAT((q31_t)(*pIn++ * 32768.0f), 16);
        *pDst++ = (q15_t)__SSAT((q31_t)(*pIn++ * 32768.0f), 16);
        *pDst++ = (q15_t)__SSAT((q31_t)(*pIn++ * 32768.0f), 16);
        *pDst++ = (q15_t)__SSAT((q31_t)(*pIn++ * 32768.0f), 16);

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* C = A * 32768 */

        /* convert from float to Q15 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 32768.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = (q15_t)(__SSAT((q31_t)(in), 16));

#else

        /* C = A * 32768 */
        /* Convert from float to q15 and then store the results in the destination buffer */
        *pDst++ = (q15_t)__SSAT((q31_t)(*pIn++ * 32768.0f), 16);

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }
#endif /* defined(RISCV_MATH_VECTOR) */
}

RISCV_DSP_ATTRIBUTE riscv_status riscv_fir_init_q15(
    riscv_fir_instance_q15 *S,
    uint16_t numTaps,
    const q15_t *pCoeffs,
    q15_t *pState,
    uint32_t blockSize)
{
    riscv_status status;

    /* Assign filter taps */
    S->numTaps = numTaps;

    /* Assign coefficient pointer */
    S->pCoeffs = pCoeffs;

    /* Clear state buffer. The size is always (blockSize + numTaps - 1) */
    memset(pState, 0, (numTaps + (blockSize - 1U)) * sizeof(q15_t));

    /* Assign state pointer */
    S->pState = pState;

    status = RISCV_MATH_SUCCESS;

    return (status);
}

__STATIC_FORCEINLINE q31_t clip_q63_to_q31(
    q63_t x)
{
    return ((q31_t)(x >> 32) != ((q31_t)x >> 31)) ? ((0x7FFFFFFF ^ ((q31_t)(x >> 63)))) : (q31_t)x;
}

/**
 * @brief Clips Q63 to Q15 values.
 */
__STATIC_FORCEINLINE q15_t clip_q63_to_q15(
    q63_t x)
{
    return ((q31_t)(x >> 32) != ((q31_t)x >> 31)) ? ((0x7FFF ^ ((q15_t)(x >> 63)))) : (q15_t)(x >> 15);
}

/**
 * @brief Clips Q31 to Q7 values.
 */
__STATIC_FORCEINLINE q7_t clip_q31_to_q7(
    q31_t x)
{
    return ((q31_t)(x >> 24) != ((q31_t)x >> 23)) ? ((0x7F ^ ((q7_t)(x >> 31)))) : (q7_t)x;
}

/**
 * @brief Clips Q31 to Q15 values.
 */
__STATIC_FORCEINLINE q15_t clip_q31_to_q15(
    q31_t x)
{
    return ((q31_t)(x >> 16) != ((q31_t)x >> 15)) ? ((0x7FFF ^ ((q15_t)(x >> 31)))) : (q15_t)x;
}

RISCV_DSP_ATTRIBUTE void riscv_fir_q15(
    const riscv_fir_instance_q15 *S,
    const q15_t *pSrc,
    q15_t *pDst,
    uint32_t blockSize)
{
    q15_t *pState = S->pState;         /* State pointer */
    const q15_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    q15_t *pStateCurnt;                /* Points to the current sample of the state */
    q15_t *px;                         /* Temporary pointer for state buffer */
    const q15_t *pb;                   /* Temporary pointer for coefficient buffer */
    q63_t acc0;                        /* Accumulators */
    uint32_t numTaps = S->numTaps;     /* Number of filter coefficients in the filter */
    uint32_t tapCnt, blkCnt;           /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    q63_t acc1, acc2, acc3; /* Accumulators */
    q31_t x0, x1, x2, c0;   /* Temporary variables to hold state and coefficient values */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64)
    uint32_t i, j;
    size_t l;
    vint16m2_t vx;
    vint64m8_t vres0m8;
    q15_t *pOut = pDst;
    /* Copy samples into state buffer */
    riscv_copy_q15(pSrc, pStateCurnt, blockSize);
    for (i = blockSize; i > 0; i -= l)
    {
        l = __riscv_vsetvl_e16m2(i);
        vx = __riscv_vle16_v_i16m2(pState, l);
        pState += l;
        vres0m8 = __riscv_vmv_v_x_i64m8(0, l);
        for (j = 0; j < numTaps; j++)
        {
            vres0m8 = __riscv_vwmacc_vx_i64m8(vres0m8, *(pCoeffs + j), __riscv_vwadd_vx_i32m4(vx, 0, l), l);
            vx = __riscv_vslide1down_vx_i16m2(vx, *(pState + j), l);
        }
        __riscv_vse16_v_i16m2(pOut, __riscv_vnclip_wx_i16m2(__riscv_vnsra_wx_i32m4(vres0m8, 15, l), 0, __RISCV_VXRM_RNU, l), l);
        pOut += l;
    }
    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

    /* Copy data */
    riscv_copy_q15(pState, pStateCurnt, numTaps - 1);
#else

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 output values simultaneously.
     * The variables acc0 ... acc3 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* Copy 4 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;

        /* Typecast q15_t pointer to q31_t pointer for state reading in q31_t */
        px = pState;

        /* Typecast q15_t pointer to q31_t pointer for coefficient reading in q31_t */
        pb = pCoeffs;

        /* Read the first two samples from the state buffer:  x[n-N], x[n-N-1] */
        x0 = read_q15x2_ia(&px);

        /* Read the third and forth samples from the state buffer: x[n-N-2], x[n-N-3] */
        x2 = read_q15x2_ia(&px);

        /* Loop over the number of taps.  Unroll by a factor of 4.
           Repeat until we've computed numTaps-(numTaps%4) coefficients. */
        tapCnt = numTaps >> 2U;

        while (tapCnt > 0U)
        {
            /* Read the first two coefficients using SIMD:  b[N] and b[N-1] coefficients */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* acc0 +=  b[N] * x[n-N] + b[N-1] * x[n-N-1] */
            acc0 = __SMLALD(x0, c0, acc0);

            /* acc2 +=  b[N] * x[n-N-2] + b[N-1] * x[n-N-3] */
            acc2 = __SMLALD(x2, c0, acc2);

            /* pack  x[n-N-1] and x[n-N-2] */
            x1 = __PKHBT(x2, x0, 0);

            /* Read state x[n-N-4], x[n-N-5] */
            x0 = read_q15x2_ia((q15_t **)&px);

            /* acc1 +=  b[N] * x[n-N-1] + b[N-1] * x[n-N-2] */
            acc1 = __SMLALDX(x1, c0, acc1);

            /* pack  x[n-N-3] and x[n-N-4] */
            x1 = __PKHBT(x0, x2, 0);

            /* acc3 +=  b[N] * x[n-N-3] + b[N-1] * x[n-N-4] */
            acc3 = __SMLALDX(x1, c0, acc3);

            /* Read coefficients b[N-2], b[N-3] */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* acc0 +=  b[N-2] * x[n-N-2] + b[N-3] * x[n-N-3] */
            acc0 = __SMLALD(x2, c0, acc0);

            /* Read state x[n-N-6], x[n-N-7] with offset */
            x2 = read_q15x2_ia((q15_t **)&px);

            /* acc2 +=  b[N-2] * x[n-N-4] + b[N-3] * x[n-N-5] */
            acc2 = __SMLALD(x0, c0, acc2);

            /* acc1 +=  b[N-2] * x[n-N-3] + b[N-3] * x[n-N-4] */
            acc1 = __SMLALDX(x1, c0, acc1);

            /* pack  x[n-N-5] and x[n-N-6] */
            x1 = __PKHBT(x2, x0, 0);

            /* acc3 +=  b[N-2] * x[n-N-5] + b[N-3] * x[n-N-6] */
            acc3 = __SMLALDX(x1, c0, acc3);

            /* Decrement tap count */
            tapCnt--;
        }

        /* If the filter length is not a multiple of 4, compute the remaining filter taps.
           This is always be 2 taps since the filter length is even. */
        if ((numTaps & 0x3U) != 0U)
        {
            /* Read last two coefficients */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* Perform the multiply-accumulates */
            acc0 = __SMLALD(x0, c0, acc0);
            acc2 = __SMLALD(x2, c0, acc2);

            /* pack state variables */
            x1 = __PKHBT(x2, x0, 0);

            /* Read last state variables */
            x0 = read_q15x2((q15_t *)px);

            /* Perform the multiply-accumulates */
            acc1 = __SMLALDX(x1, c0, acc1);

            /* pack state variables */
            x1 = __PKHBT(x0, x2, 0);

            /* Perform the multiply-accumulates */
            acc3 = __SMLALDX(x1, c0, acc3);
        }

        /* The results in the 4 accumulators are in 2.30 format. Convert to 1.15 with saturation.
           Then store the 4 outputs in the destination buffer. */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
        write_q15x4_ia(&pDst, __RV_PKBB32(__PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16), __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16)));
#else
        write_q15x2_ia(&pDst, __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16));
        write_q15x2_ia(&pDst, __PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16));
#endif /* (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

        /* Advance the state pointer by 4 to process the next group of 4 samples */
        pState = pState + 4U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy two samples into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0;

        /* Use SIMD to hold states and coefficients */
        px = pState;
        pb = pCoeffs;
        tapCnt = numTaps >> 1U;

        while (tapCnt > 0U)
        {
            acc0 += (q31_t)*px++ * *pb++;
            acc0 += (q31_t)*px++ * *pb++;

            tapCnt--;
        }

        /* The result is in 2.30 format. Convert to 1.15 with saturation.
           Then store the output in the destination buffer. */
        *pDst++ = (q15_t)(__SSAT((acc0 >> 15), 16));

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }
#endif /* defined (RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64) */
}

#define TEST_LENGTH_SAMPLES_Q31 1024
#define NUM_TAPS_Q31 32 /* Must be even */

static q31_t testInput_q31_50Hz_200Hz[TEST_LENGTH_SAMPLES_Q31] = {};

static q31_t firCoeffLP_q31[NUM_TAPS_Q31] = {};

RISCV_DSP_ATTRIBUTE void riscv_float_to_q31(
    const float32_t *pSrc,
    q31_t *pDst,
    uint32_t blockSize)
{
    uint32_t blkCnt;             /* Loop counter */
    const float32_t *pIn = pSrc; /* Source pointer */

#if defined(RISCV_MATH_VECTOR)
    blkCnt = blockSize; /* Loop counter */
    size_t l;
    vfloat32m4_t v_in;
    vint32m4_t v_out;
    for (; (l = __riscv_vsetvl_e32m4(blkCnt)) > 0; blkCnt -= l)
    {
        v_in = __riscv_vle32_v_f32m4(pIn, l);
        pIn += l;
#ifdef RISCV_MATH_ROUNDING
        v_out = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(v_in, 2147483648.0f, l), l);
#else
        v_out = __riscv_vfcvt_rtz_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(v_in, 2147483648.0f, l), l);
#endif
        __riscv_vse32_v_i32m4(pDst, v_out, l);
        pDst += l;
    }
#else
#ifdef RISCV_MATH_ROUNDING
    float32_t in;
#endif /* #ifdef RISCV_MATH_ROUNDING */

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 outputs at a time */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* C = A * 2147483648 */

        /* convert from float to Q31 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 2147483648.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = clip_q63_to_q31((q63_t)(in));

        in = (*pIn++ * 2147483648.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = clip_q63_to_q31((q63_t)(in));

        in = (*pIn++ * 2147483648.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = clip_q63_to_q31((q63_t)(in));

        in = (*pIn++ * 2147483648.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = clip_q63_to_q31((q63_t)(in));

#else

        /* C = A * 2147483648 */
        /* Convert from float to Q31 and then store the results in the destination buffer */
        *pDst++ = clip_q63_to_q31((q63_t)(*pIn++ * 2147483648.0f));
        *pDst++ = clip_q63_to_q31((q63_t)(*pIn++ * 2147483648.0f));
        *pDst++ = clip_q63_to_q31((q63_t)(*pIn++ * 2147483648.0f));
        *pDst++ = clip_q63_to_q31((q63_t)(*pIn++ * 2147483648.0f));

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* C = A * 2147483648 */

        /* convert from float to Q31 and store result in destination buffer */
#ifdef RISCV_MATH_ROUNDING

        in = (*pIn++ * 2147483648.0f);
        in += in > 0.0f ? 0.5f : -0.5f;
        *pDst++ = clip_q63_to_q31((q63_t)(in));

#else

        /* C = A * 2147483648 */
        /* Convert from float to Q31 and then store the results in the destination buffer */
        *pDst++ = clip_q63_to_q31((q63_t)(*pIn++ * 2147483648.0f));

#endif /* #ifdef RISCV_MATH_ROUNDING */

        /* Decrement loop counter */
        blkCnt--;
    }
#endif /* defined(RISCV_MATH_VECTOR) */
}

RISCV_DSP_ATTRIBUTE void riscv_fir_init_q31(
    riscv_fir_instance_q31 *S,
    uint16_t numTaps,
    const q31_t *pCoeffs,
    q31_t *pState,
    uint32_t blockSize)
{
    /* Assign filter taps */
    S->numTaps = numTaps;

    /* Assign coefficient pointer */
    S->pCoeffs = pCoeffs;

    /* Clear state buffer. The size is always (blockSize + numTaps - 1) */
    memset(pState, 0, (numTaps + (blockSize - 1U)) * sizeof(q31_t));

    /* Assign state pointer */
    S->pState = pState;
}

RISCV_DSP_ATTRIBUTE void riscv_fir_q31(
    const riscv_fir_instance_q31 *S,
    const q31_t *pSrc,
    q31_t *pDst,
    uint32_t blockSize)
{
    q31_t *pState = S->pState;         /* State pointer */
    const q31_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    q31_t *pStateCurnt;                /* Points to the current sample of the state */
    q31_t *px;                         /* Temporary pointer for state buffer */
    const q31_t *pb;                   /* Temporary pointer for coefficient buffer */
    q63_t acc0;                        /* Accumulator */
    uint32_t numTaps = S->numTaps;     /* Number of filter coefficients in the filter */
    uint32_t i, tapCnt, blkCnt;        /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    q63_t acc1, acc2; /* Accumulators */
    q31_t x0, x1, x2; /* Temporary variables to hold state values */
    q31_t c0;         /* Temporary variable to hold coefficient value */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64)
    uint32_t j;
    size_t l;
    vint32m4_t vx;
    vint64m8_t vres0m8;
    q31_t *pOut = pDst;
    /* Copy samples into state buffer */
    riscv_copy_q31(pSrc, pStateCurnt, blockSize);
    for (i = blockSize; i > 0; i -= l)
    {
        l = __riscv_vsetvl_e32m4(i);
        vx = __riscv_vle32_v_i32m4(pState, l);
        pState += l;
        vres0m8 = __riscv_vmv_v_x_i64m8(0, l);
        for (j = 0; j < numTaps; j++)
        {
            vres0m8 = __riscv_vwmacc_vx_i64m8(vres0m8, *(pCoeffs + j), vx, l);
            vx = __riscv_vslide1down_vx_i32m4(vx, *(pState + j), l);
        }
        __riscv_vse32_v_i32m4(pOut, __riscv_vnsra_wx_i32m4(vres0m8, 31, l), l);
        pOut += l;
    }
    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

    /* Copy data */
    riscv_copy_q31(pState, pStateCurnt, numTaps - 1);
#else

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 output values simultaneously.
     * The variables acc0 ... acc3 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */

    blkCnt = blockSize / 3;

    while (blkCnt > 0U)
    {
        /* Copy 3 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0;
        acc1 = 0;
        acc2 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize coefficient pointer */
        pb = pCoeffs;

        /* Read the first 2 samples from the state buffer: x[n-numTaps], x[n-numTaps-1] */
        x0 = *px++;
        x1 = *px++;

        /* Loop unrolling: process 3 taps at a time. */
        tapCnt = numTaps / 3;

        while (tapCnt > 0U)
        {
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
            q63_t c064 = read_q31x2_ia((q31_t **)&pb);
            x2 = *(px++);

            acc0 = __RV_KMADA32(acc0, __RV_PKBB32(x1, x0), c064);
            acc1 = __RV_KMADA32(acc1, __RV_PKBB32(x2, x1), c064);
            x0 = *(px++);
            acc2 = __RV_KMADA32(acc2, __RV_PKBB32(x0, x2), c064);
            c0 = *(pb++);
            x1 = *(px++);
#else
            /* Read the b[numTaps] coefficient */
            c0 = *pb;

            /* Read x[n-numTaps-2] sample */
            x2 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += ((q63_t)x0 * c0);
            acc1 += ((q63_t)x1 * c0);
            acc2 += ((q63_t)x2 * c0);

            /* Read the coefficient and state */
            c0 = *(pb + 1U);
            x0 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += ((q63_t)x1 * c0);
            acc1 += ((q63_t)x2 * c0);
            acc2 += ((q63_t)x0 * c0);

            /* Read the coefficient and state */
            c0 = *(pb + 2U);
            x1 = *(px++);

            /* update coefficient pointer */
            pb += 3U;
#endif /*  defined (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */

            /* Perform the multiply-accumulates */
            acc0 += ((q63_t)x2 * c0);
            acc1 += ((q63_t)x0 * c0);
            acc2 += ((q63_t)x1 * c0);

            /* Decrement loop counter */
            tapCnt--;
        }

        /* Loop unrolling: Compute remaining outputs */
        tapCnt = numTaps % 0x3U;

        while (tapCnt > 0U)
        {
            /* Read coefficients */
            c0 = *(pb++);

            /* Fetch 1 state variable */
            x2 = *(px++);

            /* Perform the multiply-accumulates */
            acc0 += ((q63_t)x0 * c0);
            acc1 += ((q63_t)x1 * c0);
            acc2 += ((q63_t)x2 * c0);

            /* Reuse the present sample states for next sample */
            x0 = x1;
            x1 = x2;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* Advance the state pointer by 3 to process the next group of 3 samples */
        pState = pState + 3;

        /* The result is in 2.30 format. Convert to 1.31 and store in destination buffer. */
        *pDst++ = (q31_t)(acc0 >> 31U);
        *pDst++ = (q31_t)(acc1 >> 31U);
        *pDst++ = (q31_t)(acc2 >> 31U);

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize % 0x3U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy one sample at a time into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize Coefficient pointer */
        pb = pCoeffs;

        i = numTaps;

        /* Perform the multiply-accumulates */
        do
        {
            /* acc =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0] */
            acc0 += (q63_t)*px++ * *pb++;

            i--;
        } while (i > 0U);

        /* Result is in 2.62 format. Convert to 1.31 and store in destination buffer. */
        *pDst++ = (q31_t)(acc0 >> 31U);

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }
#endif /* defined (RISCV_MATH_VECTOR) && (__RISCV_XLEN == 64) */
}

RISCV_DSP_ATTRIBUTE void riscv_fir_fast_q15(
    const riscv_fir_instance_q15 *S,
    const q15_t *pSrc,
    q15_t *pDst,
    uint32_t blockSize)
{
    q15_t *pState = S->pState;         /* State pointer */
    const q15_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    q15_t *pStateCurnt;                /* Points to the current sample of the state */
    q15_t *px;                         /* Temporary pointer for state buffer */
    const q15_t *pb;                   /* Temporary pointer for coefficient buffer */
    q31_t acc0;                        /* Accumulators */
    uint32_t numTaps = S->numTaps;     /* Number of filter coefficients in the filter */
    uint32_t tapCnt, blkCnt;           /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t acc1, acc2, acc3; /* Accumulators */
    q31_t x0, x1, x2, c0;   /* Temporary variables to hold state and coefficient values */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_VECTOR)
    uint32_t i, j;
    size_t l;
    vint16m4_t vx;
    vint32m8_t vres0m8;
    q15_t *pOut = pDst;
    /* Copy samples into state buffer */
    riscv_copy_q15(pSrc, pStateCurnt, blockSize);
    for (i = blockSize; i > 0; i -= l)
    {
        l = __riscv_vsetvl_e16m4(i);
        vx = __riscv_vle16_v_i16m4(pState, l);
        pState += l;
        vres0m8 = __riscv_vmv_v_x_i32m8(0, l);
        for (j = 0; j < numTaps; j++)
        {
            vres0m8 = __riscv_vwmacc_vx_i32m8(vres0m8, *(pCoeffs + j), vx, l);
            vx = __riscv_vslide1down_vx_i16m4(vx, *(pState + j), l);
        }
        __riscv_vse16_v_i16m4(pOut, __riscv_vnclip_wx_i16m4(__riscv_vsra_vx_i32m8(vres0m8, 15, l), 0, __RISCV_VXRM_RNU, l), l);
        pOut += l;
    }
    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;
    /* Copy data */
    riscv_copy_q15(pState, pStateCurnt, numTaps - 1);
#else

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 output values simultaneously.
     * The variables acc0 ... acc3 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* Copy 4 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;

        /* Typecast q15_t pointer to q31_t pointer for state reading in q31_t */
        px = pState;

        /* Typecast q15_t pointer to q31_t pointer for coefficient reading in q31_t */
        pb = pCoeffs;

        /* Read the first two samples from the state buffer:  x[n-N], x[n-N-1] */
        x0 = read_q15x2_ia(&px);

        /* Read the third and forth samples from the state buffer: x[n-N-2], x[n-N-3] */
        x2 = read_q15x2_ia(&px);

        /* Loop over the number of taps.  Unroll by a factor of 4.
           Repeat until we've computed numTaps-(numTaps%4) coefficients. */
        tapCnt = numTaps >> 2U;

        while (tapCnt > 0U)
        {
            /* Read the first two coefficients using SIMD:  b[N] and b[N-1] coefficients */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* acc0 +=  b[N] * x[n-N] + b[N-1] * x[n-N-1] */
            acc0 = __SMLAD(x0, c0, acc0);

            /* acc2 +=  b[N] * x[n-N-2] + b[N-1] * x[n-N-3] */
            acc2 = __SMLAD(x2, c0, acc2);

            /* pack  x[n-N-1] and x[n-N-2] */
            x1 = __PKHBT(x2, x0, 0);

            /* Read state x[n-N-4], x[n-N-5] */
            x0 = read_q15x2_ia(&px);

            /* acc1 +=  b[N] * x[n-N-1] + b[N-1] * x[n-N-2] */
            acc1 = __SMLADX(x1, c0, acc1);

            /* pack  x[n-N-3] and x[n-N-4] */
            x1 = __PKHBT(x0, x2, 0);

            /* acc3 +=  b[N] * x[n-N-3] + b[N-1] * x[n-N-4] */
            acc3 = __SMLADX(x1, c0, acc3);

            /* Read coefficients b[N-2], b[N-3] */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* acc0 +=  b[N-2] * x[n-N-2] + b[N-3] * x[n-N-3] */
            acc0 = __SMLAD(x2, c0, acc0);

            /* Read state x[n-N-6], x[n-N-7] with offset */
            x2 = read_q15x2_ia(&px);

            /* acc2 +=  b[N-2] * x[n-N-4] + b[N-3] * x[n-N-5] */
            acc2 = __SMLAD(x0, c0, acc2);

            /* acc1 +=  b[N-2] * x[n-N-3] + b[N-3] * x[n-N-4] */
            acc1 = __SMLADX(x1, c0, acc1);

            /* pack  x[n-N-5] and x[n-N-6] */
            x1 = __PKHBT(x2, x0, 0);

            /* acc3 +=  b[N-2] * x[n-N-5] + b[N-3] * x[n-N-6] */
            acc3 = __SMLADX(x1, c0, acc3);

            /* Decrement tap count */
            tapCnt--;
        }

        /* If the filter length is not a multiple of 4, compute the remaining filter taps.
           This is always be 2 taps since the filter length is even. */
        if ((numTaps & 0x3U) != 0U)
        {
            /* Read last two coefficients */
            c0 = read_q15x2_ia((q15_t **)&pb);

            /* Perform the multiply-accumulates */
            acc0 = __SMLAD(x0, c0, acc0);
            acc2 = __SMLAD(x2, c0, acc2);

            /* pack state variables */
            x1 = __PKHBT(x2, x0, 0);

            /* Read last state variables */
            x0 = read_q15x2(px);

            /* Perform the multiply-accumulates */
            acc1 = __SMLADX(x1, c0, acc1);

            /* pack state variables */
            x1 = __PKHBT(x0, x2, 0);

            /* Perform the multiply-accumulates */
            acc3 = __SMLADX(x1, c0, acc3);
        }

        /* The results in the 4 accumulators are in 2.30 format. Convert to 1.15 with saturation.
           Then store the 4 outputs in the destination buffer. */
#if defined(RISCV_MATH_DSP) && (__RISCV_XLEN == 64)
        write_q15x4_ia(&pDst, __RV_PKBB32(__PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16), __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16)));
#else
        write_q15x2_ia(&pDst, __PKHBT(__SSAT((acc0 >> 15), 16), __SSAT((acc1 >> 15), 16), 16));
        write_q15x2_ia(&pDst, __PKHBT(__SSAT((acc2 >> 15), 16), __SSAT((acc3 >> 15), 16), 16));
#endif /* (RISCV_MATH_DSP) && (__RISCV_XLEN == 64) */
        /* Advance the state pointer by 4 to process the next group of 4 samples */
        pState = pState + 4U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize & 0x3U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy two samples into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0;

        /* Use SIMD to hold states and coefficients */
        px = pState;
        pb = pCoeffs;

        tapCnt = numTaps >> 1U;

        do
        {
            acc0 += (q31_t)*px++ * *pb++;
            acc0 += (q31_t)*px++ * *pb++;

            tapCnt--;
        } while (tapCnt > 0U);

        /* The result is in 2.30 format. Convert to 1.15 with saturation.
           Then store the output in the destination buffer. */
        *pDst++ = (q15_t)(__SSAT((acc0 >> 15), 16));

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }
#endif /* defined (RISCV_MATH_VECTOR) */
}

#define multAcc_32x32_keep32_R(a, x, y) \
    a = (q31_t)(((((q63_t)a) << 32) + ((q63_t)x * y) + 0x80000000LL) >> 32)

RISCV_DSP_ATTRIBUTE void riscv_fir_fast_q31(
    const riscv_fir_instance_q31 *S,
    const q31_t *pSrc,
    q31_t *pDst,
    uint32_t blockSize)
{
    q31_t *pState = S->pState;         /* State pointer */
    const q31_t *pCoeffs = S->pCoeffs; /* Coefficient pointer */
    q31_t *pStateCurnt;                /* Points to the current sample of the state */
    q31_t *px;                         /* Temporary pointer for state buffer */
    const q31_t *pb;                   /* Temporary pointer for coefficient buffer */
    q31_t acc0;                        /* Accumulators */
    uint32_t numTaps = S->numTaps;     /* Number of filter coefficients in the filter */
    uint32_t i, tapCnt, blkCnt;        /* Loop counters */

#if defined(RISCV_MATH_LOOPUNROLL)
    q31_t acc1, acc2, acc3;   /* Accumulators */
    q31_t x0, x1, x2, x3, c0; /* Temporary variables to hold state and coefficient values */
#endif

    /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
    /* pStateCurnt points to the location where the new input data should be written */
    pStateCurnt = &(S->pState[(numTaps - 1U)]);

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 output values simultaneously.
     * The variables acc0 ... acc3 hold output values that are being computed:
     *
     *    acc0 =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0]
     *    acc1 =  b[numTaps-1] * x[n-numTaps]   + b[numTaps-2] * x[n-numTaps-1] + b[numTaps-3] * x[n-numTaps-2] +...+ b[0] * x[1]
     *    acc2 =  b[numTaps-1] * x[n-numTaps+1] + b[numTaps-2] * x[n-numTaps]   + b[numTaps-3] * x[n-numTaps-1] +...+ b[0] * x[2]
     *    acc3 =  b[numTaps-1] * x[n-numTaps+2] + b[numTaps-2] * x[n-numTaps+1] + b[numTaps-3] * x[n-numTaps]   +...+ b[0] * x[3]
     */
    blkCnt = blockSize >> 2U;

    while (blkCnt > 0U)
    {
        /* Copy 4 new input samples into the state buffer. */
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;
        *pStateCurnt++ = *pSrc++;

        /* Set all accumulators to zero */
        acc0 = 0;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize coefficient pointer */
        pb = pCoeffs;

        /* Read the first 3 samples from the state buffer:
         *  x[n-numTaps], x[n-numTaps-1], x[n-numTaps-2] */
        x0 = *px++;
        x1 = *px++;
        x2 = *px++;

        /* Loop unrolling. Process 4 taps at a time. */
        tapCnt = numTaps >> 2U;

        /* Loop over the number of taps.  Unroll by a factor of 4.
           Repeat until we've computed numTaps-4 coefficients. */
        while (tapCnt > 0U)
        {
            /* Read the b[numTaps] coefficient */
            c0 = *pb;

            /* Read x[n-numTaps-3] sample */
            x3 = *px;

            /* acc0 +=  b[numTaps] * x[n-numTaps] */
            multAcc_32x32_keep32_R(acc0, x0, c0);

            /* acc1 +=  b[numTaps] * x[n-numTaps-1] */
            multAcc_32x32_keep32_R(acc1, x1, c0);

            /* acc2 +=  b[numTaps] * x[n-numTaps-2] */
            multAcc_32x32_keep32_R(acc2, x2, c0);

            /* acc3 +=  b[numTaps] * x[n-numTaps-3] */
            multAcc_32x32_keep32_R(acc3, x3, c0);

            /* Read the b[numTaps-1] coefficient */
            c0 = *(pb + 1U);

            /* Read x[n-numTaps-4] sample */
            x0 = *(px + 1U);

            /* Perform the multiply-accumulates */
            multAcc_32x32_keep32_R(acc0, x1, c0);
            multAcc_32x32_keep32_R(acc1, x2, c0);
            multAcc_32x32_keep32_R(acc2, x3, c0);
            multAcc_32x32_keep32_R(acc3, x0, c0);

            /* Read the b[numTaps-2] coefficient */
            c0 = *(pb + 2U);

            /* Read x[n-numTaps-5] sample */
            x1 = *(px + 2U);

            /* Perform the multiply-accumulates */
            multAcc_32x32_keep32_R(acc0, x2, c0);
            multAcc_32x32_keep32_R(acc1, x3, c0);
            multAcc_32x32_keep32_R(acc2, x0, c0);
            multAcc_32x32_keep32_R(acc3, x1, c0);

            /* Read the b[numTaps-3] coefficients */
            c0 = *(pb + 3U);

            /* Read x[n-numTaps-6] sample */
            x2 = *(px + 3U);

            /* Perform the multiply-accumulates */
            multAcc_32x32_keep32_R(acc0, x3, c0);
            multAcc_32x32_keep32_R(acc1, x0, c0);
            multAcc_32x32_keep32_R(acc2, x1, c0);
            multAcc_32x32_keep32_R(acc3, x2, c0);

            /* update coefficient pointer */
            pb += 4U;
            px += 4U;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* If the filter length is not a multiple of 4, compute the remaining filter taps */
        tapCnt = numTaps % 0x4U;

        while (tapCnt > 0U)
        {
            /* Read coefficients */
            c0 = *(pb++);

            /* Fetch 1 state variable */
            x3 = *(px++);

            /* Perform the multiply-accumulates */
            multAcc_32x32_keep32_R(acc0, x0, c0);
            multAcc_32x32_keep32_R(acc1, x1, c0);
            multAcc_32x32_keep32_R(acc2, x2, c0);
            multAcc_32x32_keep32_R(acc3, x3, c0);

            /* Reuse the present sample states for next sample */
            x0 = x1;
            x1 = x2;
            x2 = x3;

            /* Decrement loop counter */
            tapCnt--;
        }

        /* The results in the 4 accumulators are in 2.30 format. Convert to 1.31
           Then store the 4 outputs in the destination buffer. */
        *pDst++ = (q31_t)(acc0 << 1);
        *pDst++ = (q31_t)(acc1 << 1);
        *pDst++ = (q31_t)(acc2 << 1);
        *pDst++ = (q31_t)(acc3 << 1);

        /* Advance the state pointer by 4 to process the next group of 4 samples */
        pState = pState + 4U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Loop unrolling: Compute remaining output samples */
    blkCnt = blockSize % 0x4U;

#else

    /* Initialize blkCnt with number of taps */
    blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
        /* Copy one sample at a time into state buffer */
        *pStateCurnt++ = *pSrc++;

        /* Set the accumulator to zero */
        acc0 = 0;

        /* Initialize state pointer */
        px = pState;

        /* Initialize Coefficient pointer */
        pb = pCoeffs;

        i = numTaps;

        /* Perform the multiply-accumulates */
        do
        {
            multAcc_32x32_keep32_R(acc0, (*px++), (*pb++));
            i--;
        } while (i > 0U);

        /* The result is in 2.30 format. Convert to 1.31
           Then store the output in the destination buffer. */
        *pDst++ = (q31_t)(acc0 << 1);

        /* Advance state pointer by 1 for the next sample */
        pState = pState + 1U;

        /* Decrement loop counter */
        blkCnt--;
    }

    /* Processing is complete.
       Now copy the last numTaps - 1 samples to the start of the state buffer.
       This prepares the state buffer for the next function call. */

    /* Points to the start of the state buffer */
    pStateCurnt = S->pState;

#if defined(RISCV_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 taps at a time */
    tapCnt = (numTaps - 1U) >> 2U;

    /* Copy data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;
        *pStateCurnt++ = *pState++;

        /* Decrement loop counter */
        tapCnt--;
    }

    /* Calculate remaining number of copies */
    tapCnt = (numTaps - 1U) & 0x3U;

#else

    /* Initialize tapCnt with number of taps */
    tapCnt = (numTaps - 1U);

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

    /* Copy remaining data */
    while (tapCnt > 0U)
    {
        *pStateCurnt++ = *pState++;

        /* Decrement the loop counter */
        tapCnt--;
    }
}
