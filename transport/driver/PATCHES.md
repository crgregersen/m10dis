# Experimental UVM patches

For the 580.178.04 experimental UVM build, apply patches in this order to a
copy of `/usr/src/nvidia-srv-580.178.04`:

1. `patches/580.178.04-glm53-diagnostic-extension.patch`
2. `patches/580.178.04-qwen38-rtx-peer-multiroute.patch`
3. `patches/580.178.04-qwen38-rtx-peer-pte-capacity.patch`
4. `patches/580.178.04-qwen38-pte-lifecycle-cleanup.patch`
5. `patches/580.178.04-qwen38-pte-reap-dead.patch`

The second patch adds the external BAR1 owner operation, bounded retained
owner/DMA route state, and requester-HAL PTE construction used by the RTX to
M10 path. It also carries the tested local cache diagnostic additions from the
same expanded source snapshot. The second patch restores the UVM ioctl header
to the expanded-source ABI with the historical 128-entry guarded PTE capacity.

The supplemental patch was applied to a source tree after the first patch;
both resulting files matched the preserved expanded source byte-for-byte:

- `nvidia-uvm/uvm_tools.c`: SHA256
  `1892741fb8237f5909b8980e6c91af39b5d8b1a90ea6a4988386f97d5ee84f46`
- `nvidia-uvm/uvm_ioctl.h`: SHA256
  `179ea1d93b5753677b84c473b9f7addfe19a1b77b13194119bd0c98c461b9b56`

The third patch raises the guarded PTE-state capacity to 512 entries and keeps
the final UVM header aligned with `abi/580.178.04/uvm_ioctl.h`. The
current vLLM route setup retains 24 mailboxes with 16 4-KiB leaves each
(384 simultaneous aliases), so the earlier 128-entry capacity was insufficient
for that setup. The ioctl structure layout is unchanged; the ABI capacity
constant is kept in sync in `abi/580.178.04/uvm_ioctl.h`.

The fourth patch releases the diagnostic's retained PTE-table references when
UVM has completed destruction of the corresponding GPU VA space. UVM has
already removed that VA space's page tables and destroyed its RM address space
at this hook, so the diagnostic entry can no longer be restored and must not
consume a global slot forever. This handles future teardown of dead VA spaces;
it does not purge already-dead entries from a currently loaded module, nor does
it discard an entry for a still-active VA space whose target mapping was
unmapped. Such active entries still require a separately validated guarded
recovery path. The patch must be built and reviewed before any module reload;
it has not been loaded into the running system.

The earlier 128-entry preserved module has SHA256
`04936c1d6972c1b69b0f43223b2a5b34bb512cca81db9a4c24c7399d9855d7ac`,
srcversion `B01DE51450215128C734350`, and vermagic
`7.0.14-m10late8 SMP preempt mod_unload modversions`. In its prior checked
RTX aggregation run, 8 owner retains, 8 DMA maps, 24 requester PTE installs,
24 exact restores, and all route cleanups passed. That run is prior evidence;
the current vLLM one-layer gate is still required.

These are temporary diagnostic changes. Do not install them through DKMS or
the initramfs. Keep the matching module under the experimental load/test/restore
workflow and preserve the stock module separately.

The fifth patch adds an explicit `REAP_DEAD` operation. It releases only
retained guarded-PTE records whose referenced GPU VA space is already in the
terminal `DEAD` state; live worker VA spaces and their mappings are untouched.
This recovers slot capacity stranded by failed RTX CUDA contexts after their UVM
VA spaces have been destroyed. It is diagnostic state cleanup only: it does not
change a PTE, DMA mapping, BAR mapping, or allocation. The operation has not
been loaded or invoked on the live system.
