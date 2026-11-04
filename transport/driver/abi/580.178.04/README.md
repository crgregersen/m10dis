# UVM diagnostic ABI overlay for the 580.178.04 sandbox

`uvm_ioctl.h` is the matching UVM ioctl header with this repository's additive
GLM53 diagnostic commands applied. `uvm_linux_ioctl.h` is the matching Linux
wrapper header. Both were captured from the source tree used to build the
validated 580.178.04 experimental UVM module; they are used only to compile the
diagnostic sandbox against that module's ABI.

The NVIDIA common headers and the remaining UVM headers are taken from
`/usr/src/nvidia-srv-580.178.04` by
`src/build_qwen38_rtx_m10_retained_roundtrip.sh`. Do not use this overlay with a
different loaded extension ABI. The module stays under the existing
experimental load/test/restore workflow.
