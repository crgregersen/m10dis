// Multi-row decode MoE (worker mode 10, 2026-11-04): R = 1..2 token rows in ONE multicast request, for the
// MTP verify step (and batch decode later). Same transport as mode 1 (PEX multicast request, results through
// remapped pages into RTX VRAM, card-leader sums); per row the arithmetic is mode 1's, so each output row is
// bit-identical to a mode-1 call on that row. Included by the worker kernels and by the RTX peer library.
//
// Request (multicast, the 4 aliased control pages 0x0000 / 0x8000 / 0x9000 / 0xA000; BF16 rows only):
//   ple_count           R (1..2)            reserved        1 (BF16 x, as Q38_MODE1_X_BF16)
//   row 0               ids 0x8020, routes 0x8048, x 0x8070 (H/2 packed BF16 words), as mode 1
//   row 1               x at Q38_MR_X1_OFF (the second half of mode 1's FP32 x field),
//                       ids / routes at Q38_MR_IDS1_OFF / Q38_MR_ROUTES1_OFF (after the control struct)
// Per-worker block at control + Q38_MR_BLOCK_OFF, written by the RTX peer open through the owner BAR1
// mapping while the workers are parked, saved and restored like the aggregation block (+0xB000):
//   reducers (card leaders 0,4,..,16 and shared workers 20..23): ret1_va = VA of the row-1 return record,
//     remapped onto the RTX (row-0 record + 3 pages; worker 23: row-0 record - 3 pages)
//   card peers: push1_va = VA remapped onto the leader's row-1 slot (slot of worker P+2 in the peer's own VA)
//   leaders: slot1_off / slot1_stride of the 3 row-1 peer slots in their own control VA
// Reducers: record r (r >= 1) at ret1_va + (r-1) * step, step = (int32) slot1_stride (+/- 0x3000).
#pragma once
#include <stdint.h>

#define Q38_MR_MODE 10u
// 48 model layers + the MTP layer, resident as layer 48 (2026-11-04)
#ifndef Q38_LAYERS_WITH_MTP
#define Q38_LAYERS_WITH_MTP 49u
#endif
#define Q38_MR_MAX_ROWS 4u
// N rows (2026-11-04 evening, MTP with 2-3 draft tokens): rows 2 and 3 have their BF16 x at 0xC000 / 0xD400
// (3 more multicast pages per card: 0xC000, 0xD000, 0xE000) and ids / routes in the free tail of page 0xA000.
#define Q38_MR_X2_OFF 0xC000u
#define Q38_MR_X3_OFF 0xD400u
#define Q38_MR_IDS2_OFF 0xA8F0u
#define Q38_MR_ROUTES2_OFF 0xA918u
#define Q38_MR_IDS3_OFF 0xA940u
#define Q38_MR_ROUTES3_OFF 0xA968u
// Card sums for rows 1..R-1: peer k pushes them packed (row r at + (r-1) * Q38_MR_PUSH_ROW_BYTES, flag word at
// + H*4 of each row) through ONE route into the leader's block Q38_MR_PEER_BASE(k).
#define Q38_MR_PUSH_ROW_BYTES 10244u
#define Q38_MR_PEER_BASE(k) ((k) == 0 ? 0x0F000u : (k) == 1 ? 0x17000u : 0x33000u)
#define Q38_MR_PEER_BLOCK_BYTES 0x8000u      // 3 rows: 30,732 B in 8 pages
#define Q38_MR_X1_OFF 0x9470u
#define Q38_MR_IDS1_OFF 0xA8A0u
#define Q38_MR_ROUTES1_OFF 0xA8C8u
#define Q38_MR_BLOCK_OFF 0xB040u
#define Q38_MR_MAGIC 0x5138524Eu          // 'N' (N-row layout); the 2-row 'M' layout is not accepted any more
#define Q38_MR_ROLE_LEADER 1u
#define Q38_MR_ROLE_PEER 2u
#define Q38_MR_ROLE_SHARED 3u
#define Q38_MR_SLOT1_OFF 0x33000u
#define Q38_MR_SLOT1_STRIDE 0x3000u
#define Q38_MR_RECORD_SHIFT 0x3000u        // row-1 return record = row-0 record +/- 3 pages
#define Q38_MR_REASON_CONFIG 32768u        // R, BF16 format, MR block, role pointers, scratch, layer
#define Q38_MR_REASON_ROWS 65536u          // an expert id or route of row 1 is invalid
struct Q38MrBlock { uint32_t magic, role, slot1_off, slot1_stride; uint64_t ret1_va, push1_va; };
static_assert(sizeof(Q38MrBlock) == 32, "multi-row block is 8 words (saved/restored as one 8-word unit)");
static_assert(Q38_MR_X1_OFF == 0x8070u + 2560u * 2u, "row-1 x follows row-0 BF16 x");
static_assert(Q38_MR_X1_OFF + 2560u * 2u == 0xA870u, "row-1 x ends where mode 1's FP32 x field ends");
static_assert(Q38_MR_ROUTES1_OFF == Q38_MR_IDS1_OFF + 40u && Q38_MR_ROUTES1_OFF + 40u <= 0xB000u,
              "row-1 ids/routes inside the 0xA000 multicast page");
static_assert(Q38_MR_SLOT1_OFF + 3u * Q38_MR_SLOT1_STRIDE == 0x3C000u, "row-1 slots end at the timing ring");
static_assert(Q38_MR_X3_OFF + 2560u * 2u <= 0xF000u, "rows 2-3 inside the pages 0xC000-0xEFFF");
static_assert(Q38_MR_ROUTES3_OFF + 40u <= 0xB000u, "rows 2-3 ids/routes inside page 0xA000");
static_assert(3u * Q38_MR_PUSH_ROW_BYTES <= Q38_MR_PEER_BLOCK_BYTES, "three packed rows fit a peer block");
static_assert(0x0F000u >= 0xF000u && 0x17000u + Q38_MR_PEER_BLOCK_BYTES <= 0x20000u &&
              0x33000u + Q38_MR_PEER_BLOCK_BYTES <= 0x3C000u, "peer blocks clear of request rows, direct record and timing ring");
__host__ __device__ constexpr unsigned q38_mr_x_off(unsigned r) { return r == 0 ? 0x8070u : r == 1 ? Q38_MR_X1_OFF : r == 2 ? Q38_MR_X2_OFF : Q38_MR_X3_OFF; }
__host__ __device__ constexpr unsigned q38_mr_ids_off(unsigned r) { return r == 0 ? 0x8020u : r == 1 ? Q38_MR_IDS1_OFF : r == 2 ? Q38_MR_IDS2_OFF : Q38_MR_IDS3_OFF; }
__host__ __device__ constexpr unsigned q38_mr_routes_off(unsigned r) { return r == 0 ? 0x8048u : r == 1 ? Q38_MR_ROUTES1_OFF : r == 2 ? Q38_MR_ROUTES2_OFF : Q38_MR_ROUTES3_OFF; }
