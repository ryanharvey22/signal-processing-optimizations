#ifndef GCFCR_OGAE_CORTEX_M_CYCLES_H
#define GCFCR_OGAE_CORTEX_M_CYCLES_H
/* Optional physical-board instrumentation. Availability/access of DWT differs
 * across Cortex-M implementations and security configurations. Check return
 * value and board documentation before accessing DWT; locked/security-restricted
 * registers may fault. QEMU counter values are NOT cycle evidence.
 */
#include <stdint.h>
static inline int ogae_dwt_enable(void) {
    volatile uint32_t *demcr = (volatile uint32_t *)0xe000edfc;
    volatile uint32_t *control = (volatile uint32_t *)0xe0001000;
    volatile uint32_t *counter = (volatile uint32_t *)0xe0001004;
    *demcr |= 1u << 24;
    if (*control & (1u << 25)) return 0; /* NOCYCCNT */
    *counter = 0;
    *control |= 1u;
    return (*control & 1u) != 0;
}
static inline uint32_t ogae_dwt_cycles(void) {
    uint32_t cycles;
    /* Compiler barriers on both sides preserve call placement without adding cycles.
     * No instruction is emitted and no registers are read or modified; the
     * memory clobber prevents surrounding work from crossing the counter read.
     */
    __asm__ volatile ("" ::: "memory");
    cycles = *(volatile uint32_t *)0xe0001004;
    __asm__ volatile ("" ::: "memory");
    return cycles;
}
#endif
