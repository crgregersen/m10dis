#pragma once
#include <cstddef>
#include <cstdint>

constexpr unsigned Q38_CPU_MAILBOX_WORKERS = 24;
constexpr unsigned Q38_CPU_MAILBOX_HIDDEN = 2560;
constexpr size_t Q38_CPU_MAILBOX_SLOT_STRIDE = 16384;
constexpr size_t Q38_CPU_MAILBOX_SLOT_BYTES = 16 + Q38_CPU_MAILBOX_HIDDEN * 4 + 4;
constexpr size_t Q38_CPU_MAILBOX_SLOTS_BYTES = Q38_CPU_MAILBOX_WORKERS * Q38_CPU_MAILBOX_SLOT_STRIDE;

// Batched MoE area (2026-11-03, worker mode 8; docs/BATCHED_MOE_EXCHANGE_RESEARCH.md), after the 24
// slots, so every slot offset is unchanged. Offsets are relative to Q38_BATCH_OFF (region base + 384 KiB).
// A worker finds the region base from its own slot pointer: cpu_mailbox_device_ptr - worker * stride.
//   request:  x [R][2560] f32 | ids [R][10] i32 | routes [R][10] f32        (written by the RTX)
//   gate:     g [R] f32                                                     (shared reducer, worker 20)
//   part[w]:  [R][2560] f32 rows + flag page (u32 seq at the page start)   (one area per worker w)
// Reducers (routed card leaders / workers without aggregation, shared worker 20) leave their reduced
// rows in their own part area; the RTX reads only the reducers' areas.
constexpr unsigned Q38_BATCH_MAX_ROWS = 64;
constexpr unsigned Q38_BATCH_K = 10;
constexpr size_t Q38_BATCH_OFF = Q38_CPU_MAILBOX_SLOTS_BYTES;
constexpr size_t Q38_BATCH_ROW_BYTES = size_t(Q38_CPU_MAILBOX_HIDDEN) * 4;                 // 10,240
constexpr size_t Q38_BATCH_REQ_X = 0;
constexpr size_t Q38_BATCH_REQ_IDS = Q38_BATCH_REQ_X + Q38_BATCH_MAX_ROWS * Q38_BATCH_ROW_BYTES;
constexpr size_t Q38_BATCH_REQ_ROUTES = Q38_BATCH_REQ_IDS + Q38_BATCH_MAX_ROWS * Q38_BATCH_K * 4;
// Request x format (2026-11-03), written by the RTX before the doorbell: 0 = FP32 rows (R*H floats at REQ_X),
// 1 = BF16 rows (R*H/2 words at REQ_X, element 2m in the low half): half the host read volume of the 24
// workers; exact, since BF16 -> FP32 widening is exact.
constexpr size_t Q38_BATCH_REQ_FMT = Q38_BATCH_REQ_ROUTES + Q38_BATCH_MAX_ROWS * Q38_BATCH_K * 4;
constexpr unsigned Q38_BATCH_FMT_FP32 = 0, Q38_BATCH_FMT_BF16 = 1;
constexpr size_t Q38_BATCH_GATE = 0xA2000;                                                  // 4-KiB aligned
constexpr size_t Q38_BATCH_PART0 = Q38_BATCH_GATE + 0x1000;
constexpr size_t Q38_BATCH_PART_ROWS_BYTES = Q38_BATCH_MAX_ROWS * Q38_BATCH_ROW_BYTES;       // 655,360
constexpr size_t Q38_BATCH_PART_STRIDE = Q38_BATCH_PART_ROWS_BYTES + 0x1000;                // + flag page
constexpr size_t Q38_BATCH_BYTES = Q38_BATCH_PART0 + Q38_CPU_MAILBOX_WORKERS * Q38_BATCH_PART_STRIDE;
constexpr size_t Q38_CPU_MAILBOX_BYTES = Q38_CPU_MAILBOX_SLOTS_BYTES + Q38_BATCH_BYTES;
static_assert(Q38_BATCH_REQ_FMT + 4 <= Q38_BATCH_GATE, "batch request overlaps the gate area");
static_assert(Q38_BATCH_GATE + Q38_BATCH_MAX_ROWS * 4 <= Q38_BATCH_PART0, "batch gate overlaps the part areas");
// Pipelined batched MoE (worker mode 9, 2026-11-03): two request slots of Q38_BATCH_PIPE_ROWS rows inside this
// area, so the RTX can publish chunk n+1 while the workers compute chunk n and combine chunk n-1 meanwhile.
// Slot s owns rows [s*PIPE_ROWS, (s+1)*PIPE_ROWS) of every per-row array (request x at the FP32 row stride, ids,
// routes, gate, part rows), the header {fmt, rows, layer, seq} at Q38_BATCH_REQ_HDR + 16*s (slot 0 word 0 is
// Q38_BATCH_REQ_FMT) and the flag / credit words at +32*s of each flag page. Slot 0 is the mode-8 layout.
constexpr unsigned Q38_BATCH_PIPE_SLOTS = 2;
constexpr unsigned Q38_BATCH_PIPE_ROWS = Q38_BATCH_MAX_ROWS / Q38_BATCH_PIPE_SLOTS;
constexpr size_t Q38_BATCH_REQ_HDR = Q38_BATCH_REQ_FMT;
constexpr size_t Q38_BATCH_HDR_BYTES = 16;
constexpr unsigned Q38_BATCH_FLAG_SLOT_WORDS = 32;
static_assert(Q38_BATCH_REQ_HDR + Q38_BATCH_PIPE_SLOTS * Q38_BATCH_HDR_BYTES <= Q38_BATCH_GATE, "pipeline headers overlap the gate area");
static_assert(Q38_BATCH_PIPE_SLOTS * Q38_BATCH_FLAG_SLOT_WORDS * 4 <= 0x1000, "pipeline flag words exceed the flag page");
// Routed worker device scratch for mode 8 (allocated by the routed service, pointer in the control block):
//   x stage [R][2560] f32 | act [R][10][32] f32 | pd [R][10][2560] f32 | expert lists
constexpr size_t Q38_BATCH_SCRATCH_X = 0;
constexpr size_t Q38_BATCH_SCRATCH_ACT = Q38_BATCH_SCRATCH_X + Q38_BATCH_MAX_ROWS * Q38_BATCH_ROW_BYTES;
constexpr size_t Q38_BATCH_SCRATCH_PD = Q38_BATCH_SCRATCH_ACT + Q38_BATCH_MAX_ROWS * Q38_BATCH_K * 32 * 4;
constexpr size_t Q38_BATCH_SCRATCH_LISTS = Q38_BATCH_SCRATCH_PD + size_t(Q38_BATCH_MAX_ROWS) * Q38_BATCH_K * Q38_BATCH_ROW_BYTES;
constexpr size_t Q38_BATCH_SCRATCH_BYTES = Q38_BATCH_SCRATCH_LISTS + 64 * 1024;
// Shared worker device scratch for mode 8 (allocated by the shared service, same control fields):
//   x [R][2560] f32 | hidden [R][160] f32 | down outputs [R][2560] f32
constexpr unsigned Q38_BATCH_SHARED_S = 160;
constexpr size_t Q38_SBATCH_SCRATCH_X = 0;
constexpr size_t Q38_SBATCH_SCRATCH_HID = Q38_SBATCH_SCRATCH_X + Q38_BATCH_MAX_ROWS * Q38_BATCH_ROW_BYTES;
constexpr size_t Q38_SBATCH_SCRATCH_DN = Q38_SBATCH_SCRATCH_HID + Q38_BATCH_MAX_ROWS * Q38_BATCH_SHARED_S * 4;
constexpr size_t Q38_SBATCH_SCRATCH_BYTES = Q38_SBATCH_SCRATCH_DN + Q38_BATCH_MAX_ROWS * Q38_BATCH_ROW_BYTES;
constexpr uint32_t Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN = 1u;

struct Q38CpuMailboxSlot {
    volatile uint32_t seq, status, worker, layer;
    float result[Q38_CPU_MAILBOX_HIDDEN];
    float gate;
};
static_assert(sizeof(Q38CpuMailboxSlot) == Q38_CPU_MAILBOX_SLOT_BYTES,
              "CPU mailbox slot ABI mismatch");
