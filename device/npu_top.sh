#!/bin/sh
# Live NPU monitor for Allwinner / VeriSilicon VIP9000 boards.
#
#   ./npu_top.sh [interval_seconds]
#
# btop and htop do not know about this NPU, and neither does anything else in
# the distro: the vipcore driver exposes no utilisation counter in sysfs.  What
# it does expose is temperature, clock, and which processes hold the device
# open, which between them answer "is the NPU being used, and is it throttling".
#
# For an actual busy percentage, run the workload under
# `tinyvoice_run --repeat N`, which reads the hardware's own inference timer
# through VIP_NETWORK_PROP_PROFILING and prints "NPU busy % of wall clock".

INTERVAL="${1:-1}"
DEVFREQ=$(ls -d /sys/class/devfreq/*npu* 2>/dev/null | head -1)

read_zone() {   # read_zone <type-substring> -> millicelsius, or empty
    for zone in /sys/class/thermal/thermal_zone*/; do
        case "$(cat "$zone/type" 2>/dev/null)" in
            *"$1"*) cat "$zone/temp" 2>/dev/null; return ;;
        esac
    done
}

holders() {     # pids with /dev/vipcore open; needs root to see other users
    for fd in /proc/[0-9]*/fd/*; do
        [ "$(readlink "$fd" 2>/dev/null)" = /dev/vipcore ] || continue
        pid=$(echo "$fd" | cut -d/ -f3)
        printf '%s(%s) ' "$(cat /proc/$pid/comm 2>/dev/null)" "$pid"
    done | sort -u
}

printf 'NPU monitor -- %s, refreshing every %ss.  Ctrl-C to stop.\n\n' \
       "$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ' | awk '{print $NF}')" "$INTERVAL"

while true; do
    npu=$(read_zone npu); cpu=$(read_zone cpub); gpu=$(read_zone gpu)
    freq=$(cat "$DEVFREQ/cur_freq" 2>/dev/null)
    gov=$(cat "$DEVFREQ/governor" 2>/dev/null)
    users=$(holders)

    printf '\r\033[K%s  NPU %s C  clk %s MHz (%s)  CPU %s C  GPU %s C  in-use: %s' \
        "$(date +%H:%M:%S)" \
        "$([ -n "$npu" ] && expr "$npu" / 1000 || echo '?')" \
        "$([ -n "$freq" ] && expr "$freq" / 1000000 || echo '?')" \
        "${gov:-?}" \
        "$([ -n "$cpu" ] && expr "$cpu" / 1000 || echo '?')" \
        "$([ -n "$gpu" ] && expr "$gpu" / 1000 || echo '?')" \
        "${users:-none}"
    sleep "$INTERVAL"
done
