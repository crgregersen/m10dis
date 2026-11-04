// Crash-safe journal of patched-UVM route records (owner retains, DMA maps, guarded PTE remaps).
//
// Every record the RTX peer library acquires in the global patched-UVM tables is appended here
// (fflush + fsync per line) BEFORE the caller can use it, and every release is appended after it
// succeeds. If the client dies, the resident daemon replays the journal (RECOVER_PEER_CLIENT ->
// q38_peer_route_tool restore) while the M10 page tables still exist. Format: one tab-separated
// record per line; see docs/RESIDENT_DAEMON_INTERFACE.md section "Route journal".
//
// Enabled only when Q38_RTX_PEER_ROUTE_JOURNAL is set; otherwise every call is a no-op, so
// programs that include the shared route code are unchanged.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

static FILE *g_q38_journal = nullptr;

static void q38_journal_line(const char *fmt, ...) {
    if (!g_q38_journal) return;
    va_list ap; va_start(ap, fmt);
    std::vfprintf(g_q38_journal, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_q38_journal);
    if (std::fflush(g_q38_journal) != 0 || fsync(fileno(g_q38_journal)) != 0)
        throw std::runtime_error("route journal write/fsync failed");
}

static void q38_journal_open_from_env() {
    if (g_q38_journal) return;
    const char *path = std::getenv("Q38_RTX_PEER_ROUTE_JOURNAL");
    if (!path || !*path) return;
    g_q38_journal = std::fopen(path, "a");
    if (!g_q38_journal) throw std::runtime_error(std::string("cannot open route journal ") + path);
    q38_journal_line("OPEN\tpid\t%d", (int)getpid());
}

template <class Uuid> static std::string q38_uuid_hex(const Uuid &u) {
    static const char *hex = "0123456789abcdef";
    std::string s;
    const unsigned char *b = reinterpret_cast<const unsigned char *>(&u);
    for (size_t i = 0; i < sizeof(Uuid); i++) { s += hex[b[i] >> 4]; s += hex[b[i] & 15]; }
    return s;
}
