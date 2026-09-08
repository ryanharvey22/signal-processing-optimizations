#!/usr/bin/env python3
"""Cross-compile Cortex-M fixtures; run supported QEMU functional models."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

TARGETS = {
    "cortex-m4": ("fpv4-sp-d16", "mps2-an386"),
    "cortex-m7": ("fpv5-sp-d16", "mps2-an500"),
    "cortex-m33": ("fpv5-sp-d16", None),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("arm-none-eabi-gcc")
    if not compiler:
        raise SystemExit("arm-none-eabi-gcc is required; cross checks cannot be silently skipped")
    qemu = shutil.which("qemu-system-arm")
    machines = subprocess.check_output([qemu, "-machine", "help"], text=True) if qemu else ""
    results = {"physical_chip_latency_measured": False, "qemu_cycle_accuracy_claim": False,
               "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
               "targets": {}}
    for cpu, (fpu, machine) in TARGETS.items():
        elf = args.out / (cpu + ".elf")
        flags = ["-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-mcpu=" + cpu,
                 "-mthumb", "-mfpu=" + fpu, "-mfloat-abi=hard",
                 "-fdata-sections", "-ffunction-sections", "-fstack-usage",
                 "-Iembedded", "-I" + str(args.generated),
                 "embedded/ogae.c", "embedded/test_inference.c", "embedded/qemu/startup.c",
                 "-nostartfiles", "--specs=rdimon.specs", "-Tembedded/qemu/linker.ld",
                 "-Wl,--gc-sections,-Map=" + str(args.out / (cpu + ".map")), "-lm", "-o", str(elf)]
        subprocess.run([compiler, *flags], check=True, timeout=120)
        size = subprocess.check_output(["arm-none-eabi-size", str(elf)], text=True)
        values = size.splitlines()[1].split()
        text, data, bss = map(int, values[:3])
        if text + data > 256 * 1024 or data + bss > 64 * 1024:
            raise RuntimeError("fixture exceeds provisional flash/RAM budget")
        row = {"compiled": True, "fpu": fpu, "flags": flags, "text_bytes": text,
               "data_bytes": data, "bss_bytes": bss, "flash_text_plus_data": text + data,
               "static_ram_data_plus_bss": data + bss, "qemu_functional_pass": False,
               "limits": "linked fixture includes golden vectors; static RAM excludes runtime stack"}
        if machine and machine in machines:
            result = subprocess.run([qemu, "-M", machine, "-nographic",
                 "-semihosting-config", "enable=on,target=native", "-kernel", str(elf)],
                 text=True, capture_output=True, timeout=45)
            row["qemu_output"] = result.stdout + result.stderr
            if result.returncode:
                raise RuntimeError(f"{cpu}: QEMU functional failure: {row['qemu_output']}")
            row["qemu_functional_pass"] = True
            row["qemu_board"] = machine
        else:
            row["emulation_limit"] = "compile only; no supported startup/board pairing configured"
        results["targets"][cpu] = row
        print(cpu, json.dumps(row), flush=True)
    (args.out / "microcontrollers.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
