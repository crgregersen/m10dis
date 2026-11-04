# Source this file, then call: q38_discover [TOPOLOGY.tsv]
# Exports the Q38_* PCI variables from qwen38_pci_discover.py (read-only sysfs scan)
# and prints the discovered routes. PCI addresses are never hardcoded in the runners.
q38_discover() {
    local disc
    disc=$(python3 "$(dirname "${BASH_SOURCE[0]}")/qwen38_pci_discover.py" --env ${1:+--topology "$1"}) \
        || { echo 'ERROR,PCI discovery failed or is ambiguous; refusing to continue' >&2; return 3; }
    eval "$disc"
    local -a up out
    read -ra up <<< "$Q38_GROUP_UPSTREAMS"; read -ra out <<< "$Q38_GROUP_OUTER_PORTS"
    local g
    for g in "${!up[@]}"; do echo "GROUP_ROUTE,group,$g,upstream,${up[$g]},outer_port,${out[$g]}"; done
    echo "RTX_ROUTE,$Q38_RTX_BDF,uuid,$Q38_RTX_UUID,behind,$Q38_RTX_OUTER_PORT,outer_switch,$Q38_OUTER_SWITCH_UPSTREAM"
}
# q38_outer_ports FIRST COUNT -> outer ports of groups FIRST..FIRST+COUNT-1
q38_outer_ports() { local -a out; read -ra out <<< "$Q38_GROUP_OUTER_PORTS"; echo "${out[@]:$1:$2}"; }
# q38_inner_ports FIRST COUNT -> inner (PEX8747 downstream) ports of those groups, 4 per group
q38_inner_ports() { local -a inn; read -ra inn <<< "$Q38_GROUP_INNER_PORTS"; echo "${inn[@]:$(($1*4)):$(($2*4))}"; }
