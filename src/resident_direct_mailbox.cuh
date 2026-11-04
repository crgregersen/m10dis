#pragma once

// The worker control and input occupy the first 128 KiB GPU leaf. The RTX
// return alias must use a separate leaf: remapping a 4 KiB subrange is not
// possible when UVM has materialized this VA using a 128 KiB leaf.
constexpr size_t Q38_DIRECT_RESULT_OFF = 0x20000;
// RM maps this entire worker allocation into BAR1. The first 64 KiB remains
// the CPU-multicast command window; this larger exposure also contains the
// aligned direct-result page used by requester-side GPU reads.
constexpr size_t Q38_WORKER_BAR_MAP_BYTES = 0x40000;
struct ResidentDirectResult {
    volatile uint32_t seq;
    volatile uint32_t status;
    volatile uint32_t exit_reason;
    volatile uint32_t ready;
    float result[2560];
    float gate;
};
static_assert(sizeof(ResidentDirectResult) == 10260);
static_assert((Q38_DIRECT_RESULT_OFF % (128u << 10)) == 0);
static_assert(Q38_DIRECT_RESULT_OFF + sizeof(ResidentDirectResult) <= (2u << 20));
static_assert(Q38_DIRECT_RESULT_OFF + (128u << 10) <= Q38_WORKER_BAR_MAP_BYTES);
