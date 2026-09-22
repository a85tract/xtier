<!-- SPDX-License-Identifier: GPL-2.0 -->
# Kernel

xTier needs a patched Linux 6.11. `build.sh` does the whole thing. It fetches
the source, patches it, builds it, installs it, and points GRUB at the result.
Reboot and you are done.

```sh
sudo ./build.sh                  # xTier only
sudo ./build.sh --stack=all      # xTier plus the baseline systems
sudo ./build.sh --slim           # much faster, see below
sudo ./build.sh --help
```

Re-running resumes rather than starting over, so an interrupted build is not a
problem.

The config is seeded from the running kernel, which on a distribution install
means building thousands of modules: **~30 GB of disk and 30-60 minutes**.
`--slim` runs `localmodconfig` first, so only the modules this machine
currently has loaded get built. That is around 12 GB and 10-15 minutes. Use it
on the machine you are going to boot. A driver that is not loaded now will not
be built into the new kernel.

Running out of disk mid-build wastes the whole build, so the script checks up
front and refuses rather than failing an hour in.

On Debian/Ubuntu, install the build dependencies first:

```sh
sudo apt-get install -y build-essential flex bison bc dwarves \
    libssl-dev libelf-dev libncurses-dev zstd curl
```

After rebooting:

```sh
uname -r                                  # 6.11.0-xtier
grep xtier_migrate_range /proc/kallsyms   # one line
ls /sys/kernel/btf/vmlinux                # needed to build the BPF program
```

## The patches

**`patches/xtier.patch`** is the only change xTier itself requires. It adds
`xtier_migrate_range()` to `mm/mempolicy.c` and exports it GPL-only, plus the
declaration in `include/linux/mempolicy.h`.

The executor module runs in kthread context and has to migrate pages belonging
to another process. Nothing suitable is exported to modules. The migration
entry points in `mm/` are internal, and `move_pages(2)` acts on the calling
task. Without this the module builds but `insmod` fails on an unresolved
symbol.

Because the export is `EXPORT_SYMBOL_GPL`, a module that links it must be GPL.
That is why `src/` and `kernel/` are GPL-2.0.

**`patches/xtier-baselines.patch`** is that plus the kernel support the
comparison systems need. Apply it only if you want to reproduce the comparison.

| Change | For |
|---|---|
| `mm/memory-tiers.c` -- node 0 as top tier, node 1 as its demotion target, node 1 given a slower abstract distance | AutoNUMA, TPP |
| `kernel/sched/fair.c` -- pins the AutoNUMA scan period | AutoNUMA, TPP |
| `mm/vmscan.c`, `kernel/sysctl.c` -- `freqtier_disable_kswapd_shrink` sysctl | HybridTier |
| `mm/htmm_*.c`, `include/linux/htmm.h` and integration points, under `CONFIG_HTMM` | Memtis |

The memory-tiers change is needed. A machine emulating CXL with a second NUMA
node has no HMAT from its BIOS. Every node then gets the same abstract
distance and lands in the same tier, which makes AutoNUMA tiering and TPP
no-ops. Without it those baselines appear to run and migrate nothing.

## Where the baselines come from

**Memtis** is included, as a patch, because it has to be. Upstream Memtis
targets Linux 5.15, and what is here is a port of its access-tracking
subsystem to 6.11. There is nowhere else to get that. Memtis is GPL-2.0, so
redistributing the port with attribution is what the licence provides for.
Upstream: <https://github.com/cosmoss-jigu/memtis>.

**HybridTier** is not included. Its kernel-side requirement is the
`freqtier_disable_kswapd_shrink` sysctl above, which is our code and is in the
patch. The runtime itself is a userspace `LD_PRELOAD` library and belongs to
its authors. Get it from
<https://github.com/kevins981/hybridtier-asplos25-artifact>. Their artifact
carries no licence file, so we do not redistribute any of it.

**AutoNUMA and TPP** are mainline. TPP has been upstream since 5.18. Both are
sysctls once the memory-tiers change is in. See `../docs/BASELINES.md`.

## Config

`build.sh` starts from the running kernel's config and then forces what
matters:

| Option | Why |
|---|---|
| `MIGRATION`, `NUMA`, `NUMA_BALANCING` | Page migration and tiering |
| `DEBUG_INFO_BTF` | The BPF program is built against a `vmlinux.h` generated from this kernel's BTF |
| `MEMORY_HOTPLUG`, `MEMORY_HOTREMOVE` | `bench/set_cxl_ratio.sh` offlines node-0 blocks to set the ratio |
| `X86_MSR` | `wrmsr -a 0x1A4 0xF` turns the hardware prefetchers off. See `docs/SETUP.md` section 5 |
| `TRANSPARENT_HUGEPAGE` | Present in the configuration we measured |
| `HTMM` | Memtis, with `--stack=all` only |
| `MODULE_SIG=n`, empty `SYSTEM_TRUSTED_KEYS` | Distribution configs point at signing keys absent from the source tree, which fails the build |

`DEBUG_INFO_BTF` needs `pahole` from the `dwarves` package. Without it the
kernel builds but `/sys/kernel/btf/vmlinux` never appears and the BPF program
cannot be compiled.

## Building by hand

If you would rather not run the script:

```sh
curl -fLO https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.11.tar.xz
tar xf linux-6.11.tar.xz && cd linux-6.11
patch -p1 < /path/to/kernel/patches/xtier.patch

cp /boot/config-$(uname -r) .config
scripts/config --enable DEBUG_INFO_BTF --enable MIGRATION \
               --enable NUMA_BALANCING --enable MEMORY_HOTREMOVE \
               --disable MODULE_SIG --set-str SYSTEM_TRUSTED_KEYS ""
scripts/config --set-str LOCALVERSION "-xtier"
make olddefconfig
make -j$(nproc) && sudo make modules_install install && sudo update-grub
```
