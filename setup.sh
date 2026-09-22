#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Installs everything xTier needs, so that `kernel/build.sh` and `make` both
# run on a machine where nothing has been set up.
#
#   sudo ./setup.sh
#
# Checks the machine first and stops if something cannot be fixed here. Run
# this before kernel/build.sh.

set -euo pipefail

SRC_DIR=/opt/linux-6.11-xtier
KVER=6.11
DO_APT=1
DO_BPFTOOL=1
DO_PYTHON=1
TORCH_CPU=1
# Kernel source plus a --slim build. A full build wants about 30 GB more.
NEED_GB=15
FULL_GB=32

usage() {
	cat <<EOF
Usage: sudo $0 [options]

  --src=DIR       Kernel source tree, shared with kernel/build.sh
                                                  (default: $SRC_DIR)
  --no-apt        Skip the distribution packages
  --no-bpftool    Skip building bpftool
  --no-python     Skip the Python packages
  --torch-cuda    Install the default (CUDA) torch wheels rather than the
                  CPU-only ones. The model is small and trains on CPU in
                  about a minute, so the CPU wheels are the default and
                  save roughly 3 GB.
  -h, --help      This message
EOF
}

for arg in "$@"; do
	case "$arg" in
		--src=*)      SRC_DIR="${arg#--src=}" ;;
		--no-apt)     DO_APT=0 ;;
		--no-bpftool) DO_BPFTOOL=0 ;;
		--no-python)  DO_PYTHON=0 ;;
		--torch-cuda) TORCH_CPU=0 ;;
		-h|--help)    usage; exit 0 ;;
		*) echo "unknown option: $arg" >&2; usage >&2; exit 1 ;;
	esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
say() { printf '\n=== %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }

FATAL=()
WARN=()
row() { printf '  %-18s %-26s %s\n' "$1" "$2" "$3"; }
fatal() { FATAL+=("$1: $2"); row "$1" "$3" "CANNOT FIX -- $2"; }
warn()  { WARN+=("$1: $2");  row "$1" "$3" "WARNING -- $2"; }

# --- check the machine -----------------------------------------------------

say "Checking this machine"

# Distribution. Package names below are Debian/Ubuntu; anything else has to be
# installed by hand, but the rest of the script still works with --no-apt.
DISTRO_ID=""; DISTRO_VER=""
if [ -r /etc/os-release ]; then
	# shellcheck disable=SC1091
	. /etc/os-release
	DISTRO_ID="${ID:-}"; DISTRO_VER="${VERSION_ID:-}"
fi
case "$DISTRO_ID" in
	ubuntu|debian) row "distro" "$DISTRO_ID $DISTRO_VER" "ok" ;;
	"") warn "distro" "unknown; package names may differ" "unidentified" ;;
	*)  if [ "$DO_APT" -eq 1 ] && ! command -v apt-get >/dev/null; then
		    fatal "distro" "not Debian/Ubuntu; install the packages by hand and re-run with --no-apt" "$DISTRO_ID $DISTRO_VER"
	    else
		    warn "distro" "not Debian/Ubuntu; package names may differ" "$DISTRO_ID $DISTRO_VER"
	    fi ;;
esac

ARCH=$(uname -m)
if [ "$ARCH" = x86_64 ]; then
	row "architecture" "$ARCH" "ok"
else
	fatal "architecture" "xTier is x86_64 only (PEBS, and the BPF program targets x86)" "$ARCH"
fi

NODES=$(find /sys/devices/system/node -maxdepth 1 -name 'node[0-9]*' 2>/dev/null | wc -l)
if [ "$NODES" -ge 2 ]; then
	row "NUMA nodes" "$NODES" "ok"
else
	fatal "NUMA nodes" "xTier places pages between two tiers and needs at least 2 nodes" "$NODES"
fi

VENDOR=$(awk -F': ' '/^vendor_id/ {print $2; exit}' /proc/cpuinfo 2>/dev/null || true)
if [ "$VENDOR" = GenuineIntel ]; then
	row "CPU vendor" "$VENDOR" "ok"
else
	warn "CPU vendor" "the PEBS event encodings are Intel; elsewhere they sample something else" "${VENDOR:-unknown}"
fi

if grep -qm1 ' hypervisor' /proc/cpuinfo 2>/dev/null; then
	warn "virtualisation" "PEBS is usually unavailable in a VM; xTier would fall back to an imprecise event" "virtual machine"
else
	row "virtualisation" "bare metal" "ok"
fi

if grep -qm1 ' dtes64' /proc/cpuinfo 2>/dev/null; then
	row "debug store" "dtes64" "ok"
else
	warn "debug store" "no dtes64 flag, so PEBS is probably unavailable" "absent"
fi

SB="disabled"
if command -v mokutil >/dev/null 2>&1; then
	mokutil --sb-state 2>/dev/null | grep -qi enabled && SB="enabled" || true
elif ls /sys/firmware/efi/efivars/SecureBoot-* >/dev/null 2>&1; then
	od -An -t u1 /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null \
		| awk '{ if ($NF == 1) exit 0; exit 1 }' && SB="enabled" || true
fi
if [ "$SB" = enabled ]; then
	warn "Secure Boot" "the kernel is built unsigned (MODULE_SIG=n) and the module will not load" "enabled"
else
	row "Secure Boot" "disabled" "ok"
fi

AVAIL_GB=$(df -BG --output=avail "$(dirname "$SRC_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
if [ -z "${AVAIL_GB:-}" ]; then
	warn "disk" "could not determine free space at $(dirname "$SRC_DIR")" "unknown"
elif [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
	fatal "disk" "need about ${NEED_GB}G at $(dirname "$SRC_DIR") for the source and a --slim build" "${AVAIL_GB}G free"
elif [ "$AVAIL_GB" -lt "$FULL_GB" ]; then
	warn "disk" "enough for kernel/build.sh --slim, not for a full build" "${AVAIL_GB}G free"
else
	row "disk" "${AVAIL_GB}G free" "ok"
fi

# PEP 668 marks the system Python as externally managed, which makes a plain
# `pip install --user` refuse. Ubuntu 24.04 and Debian 12 do; 22.04 does not.
PIP_FLAGS="--user --quiet"
if ls /usr/lib/python3*/EXTERNALLY-MANAGED >/dev/null 2>&1; then
	if python3 -m pip install --help 2>/dev/null | grep -q "break-system-packages"; then
		PIP_FLAGS="$PIP_FLAGS --break-system-packages"
		row "python packaging" "externally managed" "will pass --break-system-packages"
	else
		warn "python packaging" "externally managed but this pip cannot override it; install the ml/ and figures/ requirements into a venv by hand" "externally managed"
		DO_PYTHON=0
	fi
else
	row "python packaging" "not externally managed" "ok"
fi

if [ ${#FATAL[@]} -gt 0 ]; then
	echo
	echo "Cannot continue:" >&2
	for f in "${FATAL[@]}"; do echo "  - $f" >&2; done
	exit 1
fi
if [ ${#WARN[@]} -gt 0 ]; then
	echo
	echo "Continuing, but note:"
	for w in "${WARN[@]}"; do echo "  - $w"; done
fi

# --- distribution packages -------------------------------------------------

if [ "$DO_APT" -eq 1 ]; then
	command -v apt-get >/dev/null || {
		echo "no apt-get: install the equivalents by hand, then re-run with --no-apt." >&2
		exit 1
	}

	say "Installing packages"
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq

	PKGS=(build-essential flex bison bc dwarves
	      libssl-dev libelf-dev libncurses-dev zstd curl ca-certificates
	      clang llvm zlib1g-dev pkg-config libbpf-dev
	      libcap-dev binutils-dev
	      numactl msr-tools cgroup-tools
	      python3 python3-pip)

	# Names drift between releases, so install what this release actually has
	# and report the rest rather than failing the whole transaction.
	WANT=(); SKIP=()
	for p in "${PKGS[@]}"; do
		cand=$(apt-cache policy "$p" 2>/dev/null | awk '/Candidate:/{print $2}')
		if [ -n "$cand" ] && [ "$cand" != "(none)" ]; then WANT+=("$p"); else SKIP+=("$p"); fi
	done
	if [ ${#SKIP[@]} -gt 0 ]; then
		echo "not available on $DISTRO_ID $DISTRO_VER, skipping: ${SKIP[*]}"
	fi
	apt-get install -y --no-install-recommends "${WANT[@]}"
	echo "ok"
fi

# --- kernel source ---------------------------------------------------------

# bpftool is built from the kernel tree, so fetch it here if it is not already
# present. kernel/build.sh reuses the same directory and patches it in place.
if [ "$DO_BPFTOOL" -eq 1 ] && [ ! -f "$SRC_DIR/Makefile" ]; then
	say "Fetching Linux $KVER"
	mkdir -p "$(dirname "$SRC_DIR")"
	WORK=$(mktemp -d)
	trap 'rm -rf "$WORK"' EXIT
	curl -fL --progress-bar -o "$WORK/linux-$KVER.tar.xz" \
		"https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-$KVER.tar.xz"
	tar -C "$WORK" -xf "$WORK/linux-$KVER.tar.xz"
	rm -rf "$SRC_DIR"
	mv "$WORK/linux-$KVER" "$SRC_DIR"
fi

# --- bpftool ---------------------------------------------------------------

# The distribution package is tied to the distribution kernel: on Ubuntu 22.04
# it is v5.15, which cannot read a 6.11 BTF and fails the vmlinux.h step with
# "failed to load BTF from /sys/kernel/btf/vmlinux: Invalid argument". Building
# from the tree we are going to boot keeps the two in step.
if [ "$DO_BPFTOOL" -eq 1 ]; then
	say "Building bpftool from $SRC_DIR"
	make -C "$SRC_DIR/tools/bpf/bpftool" -j"$(nproc)" >/dev/null
	install -m755 "$SRC_DIR/tools/bpf/bpftool/bpftool" /usr/local/bin/bpftool
	hash -r 2>/dev/null || true
	/usr/local/bin/bpftool version
fi

# --- python ----------------------------------------------------------------

if [ "$DO_PYTHON" -eq 1 ]; then
	say "Installing Python packages"
	# --user, so the workflow runs as an ordinary user rather than root.
	REAL_USER="${SUDO_USER:-root}"
	PIP="python3 -m pip install $PIP_FLAGS"
	if [ "$TORCH_CPU" -eq 1 ]; then
		sudo -u "$REAL_USER" $PIP --index-url https://download.pytorch.org/whl/cpu torch
		sudo -u "$REAL_USER" $PIP pandas numpy scikit-learn
	else
		sudo -u "$REAL_USER" $PIP -r "$HERE/ml/requirements.txt"
	fi
	sudo -u "$REAL_USER" $PIP -r "$HERE/figures/requirements.txt"
	echo "ok"
fi

# --- verify ----------------------------------------------------------------

say "Verifying"

MISSING=()
for t in gcc make flex bison bc curl tar xz patch pahole clang llvm-strip \
         bpftool numactl wrmsr python3; do
	command -v "$t" >/dev/null || MISSING+=("$t")
done
[ -e /usr/include/openssl/ssl.h ] || MISSING+=("libssl-dev")
[ -e /usr/include/libelf.h ] || MISSING+=("libelf-dev")
[ -e /usr/include/zlib.h ] || MISSING+=("zlib1g-dev")

if [ ${#MISSING[@]} -gt 0 ]; then
	echo "still missing: ${MISSING[*]}" >&2
	exit 1
fi
row "tools and headers" "all present" "ok"

# Whether this machine can actually deliver PEBS samples, rather than whether
# it looks like it should. xTier opens exactly this event.
PEBS_STATE="unavailable"
if command -v gcc >/dev/null; then
	PROBE=$(mktemp -d)
	cat > "$PROBE/p.c" <<'EOF'
#define _GNU_SOURCE
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
int main(void) {
	struct perf_event_attr pe;
	memset(&pe, 0, sizeof pe);
	pe.type = PERF_TYPE_RAW;
	pe.size = sizeof pe;
	pe.config = 0x81D0;            /* MEM_UOPS_RETIRED.ALL_LOADS */
	pe.freq = 1;
	pe.sample_freq = 1000;
	pe.sample_type = PERF_SAMPLE_ADDR;
	pe.precise_ip = 2;             /* PEBS */
	pe.exclude_kernel = 1;
	pe.disabled = 1;
	int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
	if (fd < 0) return 1;
	close(fd);
	return 0;
}
EOF
	if gcc -O0 -o "$PROBE/p" "$PROBE/p.c" 2>/dev/null && "$PROBE/p"; then
		PEBS_STATE="available"
	fi
	rm -rf "$PROBE"
fi

if [ "$PEBS_STATE" = available ]; then
	row "PEBS" "event 0x81D0 opens" "ok"
else
	row "PEBS" "event 0x81D0 refused" "WARNING -- see below"
	cat >&2 <<EOF

  This machine will not give xTier precise memory samples. It builds and runs,
  but falls back to an imprecise event and the addresses it places on are not
  the ones the workload touched, so any measurement is meaningless. Usual
  causes: a virtual machine, a non-Intel CPU, or PEBS disabled in firmware.
EOF
fi

cat <<EOF

=== Done

Next:

    sudo kernel/build.sh --slim     # build and install the kernel, then reboot
    make bootstrap-model            # placeholder weights, so the loader builds
    make MODEL_HDR=\$PWD/ml/models/mlp_q8_bootstrap.h

Then collect features, train a real model, and rebuild against it. See
docs/SETUP.md and ml/README.md.
EOF
