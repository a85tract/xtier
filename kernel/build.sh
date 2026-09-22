#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# Build and install the kernel xTier needs, starting from nothing.
#
#   sudo ./build.sh                 # xTier only
#   sudo ./build.sh --stack=all     # xTier plus the baseline systems
#
# Fetches Linux 6.11, applies our patches, sets the required config, builds,
# installs, and points GRUB at the result. When it finishes, reboot.
#
# Safe to re-run: it skips the download and the patch step if they are already
# done, so an interrupted build resumes rather than starting over.

set -euo pipefail

STACK=xtier
SRC_DIR=/opt/linux-6.11-xtier
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DO_INSTALL=1
DO_GRUB=1
SLIM=0

usage() {
	cat <<EOF
Usage: sudo $0 [options]

  --stack=xtier   Only the xTier patch. This is what xTier needs.   (default)
  --stack=all     xTier plus kernel support for the baseline systems
                  (AutoNUMA/TPP tier fixes, the kswapd-shrink sysctl, and the
                  Memtis access-tracking subsystem under CONFIG_HTMM).

  --src=DIR       Where to unpack and build       (default: $SRC_DIR)
  --jobs=N        Parallel build jobs             (default: $JOBS)
  --slim          Build only the modules this machine currently has loaded,
                  via localmodconfig. Cuts the build from ~30 GB and an hour
                  to ~10 GB and 10-15 minutes. Run it on the machine you are
                  going to boot: a driver that is not loaded now will not be
                  built, and if you need it later you will have to rebuild.
  --no-install    Build only; do not install or touch GRUB
  --no-grub       Install the kernel but leave the boot order alone
  -h, --help      This message

A full distribution config builds thousands of modules: ~30 GB of disk and
30-60 minutes. See --slim if that is more than you want to spend.
EOF
}

for arg in "$@"; do
	case "$arg" in
		--stack=xtier|--stack=all) STACK="${arg#--stack=}" ;;
		--src=*)      SRC_DIR="${arg#--src=}" ;;
		--jobs=*)     JOBS="${arg#--jobs=}" ;;
		--slim)       SLIM=1 ;;
		--no-install) DO_INSTALL=0; DO_GRUB=0 ;;
		--no-grub)    DO_GRUB=0 ;;
		-h|--help)    usage; exit 0 ;;
		*) echo "unknown option: $arg" >&2; usage >&2; exit 1 ;;
	esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KVER=6.11
TARBALL="linux-$KVER.tar.xz"
URL="https://cdn.kernel.org/pub/linux/kernel/v6.x/$TARBALL"

case "$STACK" in
	xtier) PATCH="$HERE/patches/xtier.patch";           LOCALVERSION="-xtier" ;;
	all)   PATCH="$HERE/patches/xtier-baselines.patch"; LOCALVERSION="-xtier-baselines" ;;
esac
[ -f "$PATCH" ] || { echo "missing patch: $PATCH" >&2; exit 1; }

say() { printf '\n=== %s\n' "$*"; }

# --- prerequisites ---------------------------------------------------------

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }

say "Checking prerequisites"

MISSING=()
for t in gcc make flex bison bc curl tar xz patch; do
	command -v "$t" >/dev/null || MISSING+=("$t")
done
# pahole generates the BTF that the BPF program's vmlinux.h is built from.
command -v pahole >/dev/null || MISSING+=("pahole (package: dwarves)")
[ -e /usr/include/openssl/ssl.h ] || MISSING+=("libssl-dev")
{ [ -e /usr/include/libelf.h ] || [ -e /usr/include/elfutils/libelf.h ]; } || MISSING+=("libelf-dev")

if [ ${#MISSING[@]} -gt 0 ]; then
	echo "missing: ${MISSING[*]}" >&2
	echo >&2
	echo "On Debian/Ubuntu:" >&2
	echo "  apt-get install -y build-essential flex bison bc dwarves \\" >&2
	echo "      libssl-dev libelf-dev libncurses-dev zstd curl" >&2
	exit 1
fi

# A full distribution config expands to thousands of modules; --slim cuts that
# to what is currently loaded. Running out of disk mid-build wastes the whole
# build, so refuse up front rather than failing an hour in.
if [ "$SLIM" -eq 1 ]; then NEED_GB=12; else NEED_GB=30; fi
AVAIL_GB=$(df -BG --output=avail "$(dirname "$SRC_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
	echo "only ${AVAIL_GB}G free at $(dirname "$SRC_DIR"); this build wants ~${NEED_GB}G" >&2
	[ "$SLIM" -eq 0 ] && echo "try --slim, which needs about 12G" >&2
	exit 1
fi
echo "ok"

# --- fetch -----------------------------------------------------------------

if [ -f "$SRC_DIR/.xtier-patched-$STACK" ]; then
	say "Source already prepared at $SRC_DIR (stack: $STACK)"
else
	if [ -d "$SRC_DIR" ] && [ -f "$SRC_DIR/Makefile" ]; then
		say "Reusing existing tree at $SRC_DIR"
	else
		say "Fetching Linux $KVER"
		mkdir -p "$(dirname "$SRC_DIR")"
		WORK=$(mktemp -d)
		trap 'rm -rf "$WORK"' EXIT
		curl -fL --progress-bar -o "$WORK/$TARBALL" "$URL"
		tar -C "$WORK" -xf "$WORK/$TARBALL"
		rm -rf "$SRC_DIR"
		mv "$WORK/linux-$KVER" "$SRC_DIR"
	fi

	say "Applying $(basename "$PATCH")"
	cd "$SRC_DIR"
	patch -p1 --forward --dry-run < "$PATCH" >/dev/null \
		|| { echo "patch does not apply to this tree -- delete $SRC_DIR and re-run" >&2; exit 1; }
	patch -p1 --forward < "$PATCH"
	touch ".xtier-patched-$STACK"
fi

# --- configure -------------------------------------------------------------

cd "$SRC_DIR"
say "Configuring"

if [ ! -f .config ]; then
	if [ -f "/boot/config-$(uname -r)" ]; then
		echo "starting from the running kernel's config"
		cp "/boot/config-$(uname -r)" .config
	else
		echo "no running-kernel config found; starting from defconfig"
		make defconfig >/dev/null
	fi
fi

# Required by xTier.
scripts/config --enable  MIGRATION
scripts/config --enable  NUMA
scripts/config --enable  NUMA_BALANCING
scripts/config --enable  TRANSPARENT_HUGEPAGE
# The BPF program is compiled against a vmlinux.h generated from this kernel's
# BTF, so the BTF has to exist.
scripts/config --enable  DEBUG_INFO
scripts/config --enable  DEBUG_INFO_BTF
# bench/set_cxl_ratio.sh sets the DRAM:CXL ratio by offlining node-0 blocks.
scripts/config --enable  MEMORY_HOTPLUG
scripts/config --enable  MEMORY_HOTREMOVE
# docs/SETUP.md section 3 turns the prefetchers off with wrmsr, which needs
# /dev/cpu/*/msr. localmodconfig drops it unless msr is already loaded.
scripts/config --enable  X86_MSR

# Distribution configs point at signing keys that are not in the source tree,
# which fails the build. We are not shipping signed modules.
scripts/config --disable MODULE_SIG
scripts/config --disable SECURITY_LOCKDOWN_LSM
scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
# Ubuntu ships this on and it inflates the build with no benefit here.
scripts/config --disable DEBUG_INFO_BTF_MODULES

if [ "$STACK" = all ]; then
	scripts/config --enable HTMM
else
	scripts/config --disable HTMM
fi

scripts/config --set-str LOCALVERSION "$LOCALVERSION"
scripts/config --disable LOCALVERSION_AUTO

make olddefconfig >/dev/null

if [ "$SLIM" -eq 1 ]; then
	echo "trimming to currently-loaded modules"
	# Answers the prompts for symbols localmodconfig cannot infer. yes(1)
	# exits 141 on SIGPIPE, so judge make's status rather than the pipeline's.
	if ! ( set +o pipefail; yes "" | make localmodconfig >/dev/null 2>&1 ); then
		echo "localmodconfig failed" >&2
		exit 1
	fi
	# localmodconfig can drop options we need, so reassert and re-resolve.
	scripts/config --enable MIGRATION --enable NUMA_BALANCING \
		--enable DEBUG_INFO_BTF --enable MEMORY_HOTPLUG --enable MEMORY_HOTREMOVE \
		--enable X86_MSR
	[ "$STACK" = all ] && scripts/config --enable HTMM
	make olddefconfig >/dev/null
fi

for opt in MIGRATION NUMA_BALANCING DEBUG_INFO_BTF MEMORY_HOTREMOVE X86_MSR; do
	grep -q "^CONFIG_$opt=y" .config || { echo "CONFIG_$opt did not stick" >&2; exit 1; }
done
if [ "$STACK" = all ]; then
	grep -q "^CONFIG_HTMM=y" .config || { echo "CONFIG_HTMM did not stick" >&2; exit 1; }
fi

RELEASE=$(make -s kernelrelease)
echo "will build: $RELEASE"

# --- build -----------------------------------------------------------------

say "Building with -j$JOBS (30-60 minutes)"
make -j"$JOBS"

if [ "$DO_INSTALL" -eq 0 ]; then
	say "Built $RELEASE in $SRC_DIR. Not installing (--no-install)."
	exit 0
fi

# --- install ---------------------------------------------------------------

say "Installing modules and kernel"
make -j"$JOBS" modules_install
make install

# --- grub ------------------------------------------------------------------

if [ "$DO_GRUB" -eq 1 ]; then
	say "Setting $RELEASE as the default boot entry"

	# grub-set-default writes saved_entry, which GRUB reads only when
	# GRUB_DEFAULT=saved. Distributions ship GRUB_DEFAULT=0, so without this
	# the machine silently comes back on the old kernel.
	DEFAULT_GRUB=/etc/default/grub
	if [ -f "$DEFAULT_GRUB" ] && ! grep -q '^GRUB_DEFAULT=saved' "$DEFAULT_GRUB"; then
		echo "setting GRUB_DEFAULT=saved in $DEFAULT_GRUB"
		cp -n "$DEFAULT_GRUB" "$DEFAULT_GRUB.xtier-backup" || true
		if grep -q '^GRUB_DEFAULT=' "$DEFAULT_GRUB"; then
			sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' "$DEFAULT_GRUB"
		else
			echo 'GRUB_DEFAULT=saved' >> "$DEFAULT_GRUB"
		fi
	fi

	if command -v update-grub >/dev/null; then
		update-grub
	elif command -v grub2-mkconfig >/dev/null; then
		grub2-mkconfig -o /boot/grub2/grub.cfg
	else
		grub-mkconfig -o /boot/grub/grub.cfg
	fi

	# Not `ls a b | head -1`: one path is always absent, so under pipefail
	# the non-zero ls would end the script here via set -e.
	GRUB_CFG=
	for c in /boot/grub/grub.cfg /boot/grub2/grub.cfg; do
		if [ -f "$c" ]; then GRUB_CFG="$c"; break; fi
	done

	# Non-default kernels sit inside a submenu, so the id grub-set-default
	# wants is "<submenu-id>><entry-id>". Pull both out with sed rather than
	# awk: Ubuntu ships mawk, which has no 3-argument match(). No match is a
	# normal outcome, hence `|| true` on each grep.
	ID_RE='s/.*menuentry_id_option .\([^'"'"']*\).*/\1/p'
	SUBMENU=
	ENTRY=
	if [ -n "$GRUB_CFG" ]; then
		SUBMENU=$( { grep -m1 '^submenu ' "$GRUB_CFG" || true; } | sed -n "$ID_RE")
		ENTRY=$( { grep "^[[:space:]]*menuentry .*$RELEASE" "$GRUB_CFG" || true; } \
			| { grep -v recovery || true; } | head -1 | sed -n "$ID_RE")
	fi
	if [ -n "$ENTRY" ] && [ -n "$SUBMENU" ]; then
		ENTRY="$SUBMENU>$ENTRY"
	fi

	if [ -n "$ENTRY" ]; then
		grub-set-default "$ENTRY" 2>/dev/null || grub2-set-default "$ENTRY" 2>/dev/null || true
		echo "default boot entry: $ENTRY"
	else
		echo "could not identify the menu entry for $RELEASE." >&2
		echo "Select it by hand at the boot menu, or set GRUB_DEFAULT in" >&2
		echo "/etc/default/grub and re-run update-grub." >&2
	fi
fi

cat <<EOF

=== Done

Built and installed: $RELEASE
Stack:               $STACK
Source tree:         $SRC_DIR

Reboot, then check:

    uname -r                                  # expect $RELEASE
    grep xtier_migrate_range /proc/kallsyms   # expect one line
    ls /sys/kernel/btf/vmlinux                # needed to build the BPF program
EOF

if [ "$STACK" = all ]; then
	echo "    ls /sys/kernel/mm/htmm/                   # Memtis tunables"
fi

cat <<EOF

Then build xTier itself:

    cd $HERE/.. && make MODEL_HDR=\$PWD/ml/models/<your-model>.h

See docs/SETUP.md.
EOF
