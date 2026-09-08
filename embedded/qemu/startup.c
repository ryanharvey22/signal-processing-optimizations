/* Minimal Cortex-M test startup for the QEMU MPS2-AN386 memory map.
 * This is a functional semihosting harness, not production board startup.
 */
#include <stdint.h>
#include <stdlib.h>

extern uint32_t _stack_top, _data_load, _data_start, _data_end, _bss_start, _bss_end;
extern void initialise_monitor_handles(void);
extern int main(void);

static void fault(void) { for (;;) {} }
void Reset_Handler(void) {
    uint32_t *source = &_data_load;
    uint32_t *target;
    /* CPACR grants full access to the single-precision FPU. The following
     * assembly barriers complete that system-register update before any hard-
     * float C/library call. DSB/ISB have no operands; the memory clobber stops
     * the compiler from moving later memory operations above this boundary.
     */
    *(volatile uint32_t *)0xe000ed88u |= 0x00f00000u;
    __asm__ volatile ("dsb\n\tisb" ::: "memory");
    for (target = &_data_start; target < &_data_end; ++target) *target = *source++;
    for (target = &_bss_start; target < &_bss_end; ++target) *target = 0;
    initialise_monitor_handles();
    exit(main());
}

__attribute__((section(".vectors"), used))
const uintptr_t vectors[] = {
    (uintptr_t)&_stack_top, (uintptr_t)Reset_Handler,
    (uintptr_t)fault, (uintptr_t)fault, (uintptr_t)fault, (uintptr_t)fault,
    (uintptr_t)fault, 0, 0, 0, 0, (uintptr_t)fault, (uintptr_t)fault,
    0, (uintptr_t)fault, (uintptr_t)fault
};
