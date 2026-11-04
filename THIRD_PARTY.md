# Third-party code and licenses

The MIT license in `LICENSE` covers the code written for this project. These parts are derived from
other projects and keep their original licenses:

| Path | Origin | License |
|---|---|---|
| `kernel/m10-late-enable-quirks.patch` | patch to `drivers/pci/quirks.c` of the Linux kernel | GPL-2.0 |
| `transport/driver/patches/*.patch`, `transport/driver/abi/` | patches to and headers from the NVIDIA 580.178.04 kernel modules (`nvidia-uvm`) | dual MIT / GPL-2.0, as the NVIDIA kernel module source |
| `transport/rtx_mc_dma/` | Linux kernel module | GPL-2.0 (declared by `MODULE_LICENSE`) |
| `transport/driver/m10_qwen_mc_wc6.c` | Linux kernel module | GPL-2.0 (declared by `MODULE_LICENSE`) |
| `fan/drivers/nct6687/` | nct6687d by Frederic Boltz, based on nct6683 by Guenter Roeck, patched for this board | GPL-2.0-or-later |
| `fan/bin/m10-fancontrol` | lm-sensors `fancontrol` (Marius Reiner, Jean Delvare), one constant changed | GPL-2.0-or-later |
| `fan/drivers/arctic-fan-controller/` | ARCTIC fan controller hwmon driver, backported | GPL-2.0-or-later OR BSD-2-Clause (see its patch 0003) |
| `src/rm_ioctl_trace_preload.c` | written for this project | MIT |

Not included: model weights (Qwen3.8-Flash-Next-FP8, download separately under its own license),
vLLM and PyTorch (installed from PyPI, `requirements-vllm.txt`), benchmark datasets (LiveCodeBench,
MMLU-Redux, MMLU; downloaded by the user, see `docs/BENCHMARKS.md`).
