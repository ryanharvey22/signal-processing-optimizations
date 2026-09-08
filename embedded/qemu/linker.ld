/* Deliberately restrict the functional fixture to the provisional 256 KiB
 * flash / 64 KiB SRAM budget. Real boards need their own startup/linker script.
 */
ENTRY(Reset_Handler)
MEMORY {
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 256K
    RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}
_stack_top = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
    .vectors : { KEEP(*(.vectors)) } > FLASH
    .text : { *(.text*) *(.rodata*) KEEP(*(.init)) KEEP(*(.fini)) } > FLASH
    .ARM.extab : { *(.ARM.extab*) } > FLASH
    .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
    .data : {
        . = ALIGN(4); _data_start = .; *(.data*) . = ALIGN(4); _data_end = .;
    } > RAM AT > FLASH
    _data_load = LOADADDR(.data);
    .bss (NOLOAD) : {
        . = ALIGN(8); _bss_start = .; *(.bss*) *(COMMON) . = ALIGN(8); _bss_end = .;
    } > RAM
    end = .;
    _end = .;
    __end__ = .;
    ASSERT(_bss_end + 8192 <= _stack_top, "Less than 8 KiB remains for test stack/semihosting heap")
}
