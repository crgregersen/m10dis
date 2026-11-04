# Transport components

| Path | Content |
|---|---|
| `driver/patches/` | five patches to the NVIDIA 580.178.04 `nvidia-uvm` module (guarded PTE remapping, external BAR1 owner, record lifecycle); order and history in `driver/PATCHES.md` |
| `driver/abi/580.178.04/` | the patched UVM ioctl headers the user-space tools compile against |
| `driver/build-uvm.sh` | builds the patched module (expected srcversion `38263E1067A39300ABB966D`) |
| `driver/swap-uvm.sh` | swaps it in for the session, with fallback to the original |
| `driver/m10_qwen_mc_wc6.c` | helper module: bounded write-combining mappings of the six multicast apertures (built by the daemon start) |
| `rtx_mc_dma/` | helper module: IOMMU/DMA translations of the multicast windows for the RTX (built by the peer run) |

How these fit together: [docs/TRANSPORT.md](../docs/TRANSPORT.md).
