<!-- SPDX-License-Identifier: Apache-2.0 -->
# GAPBS

The [GAP Benchmark Suite](https://github.com/sbeamer/gapbs) is a set of graph
kernels with large, irregular working sets. That is what makes them useful
here. We used `bc`, `bfs`, `pr` and `tc`.

Nothing is vendored. Get it upstream:

```sh
git clone --depth 1 https://github.com/sbeamer/gapbs "$WORKLOAD_DIR/gapbs"
cd "$WORKLOAD_DIR/gapbs"
make
```

## Running

`-g N` generates a Kronecker graph with `2^N` vertices averaging 16 edges each,
in-process at startup, so the reported wall time includes graph construction.

```sh
numactl --cpunodebind=0 --preferred=1 ./bc  -g 26 -n 3
numactl --cpunodebind=0 --preferred=1 ./pr  -g 26 -n 3 -i 1000
numactl --cpunodebind=0 --preferred=1 ./tc  -g 26 -n 1
numactl --cpunodebind=0 --preferred=1 ./bfs -g 27 -n 30
```

Scale 26 is roughly a 30 GB working set. BFS itself is fast enough that graph
construction dominates, so it runs at scale 27 to spend enough time in the
steady state to be worth sampling.

Generating a graph in-process every run costs minutes. To generate once and
reuse, build the converter and load from a file:

```sh
g++ -std=c++11 -O3 -fopenmp -o converter src/converter.cc
./converter -g 26 -b "$WORKLOAD_DIR/kron_26.sg"
./bc -f "$WORKLOAD_DIR/kron_26.sg" -n 3
```

You may want a second graph that is statistically like the first but not
identical, to check that a model has not memorised one particular layout. To
get one, change `kRandSeed` in `src/util.h` in a *copy* of the tree and
rebuild the converter there.

## Collecting features

No script here. The pattern is the one in `../README.md`. Start the kernel with
`--preferred=1`, attach the profiler in `collect` mode against its PID, and let
it run. A sampling period of 2000 works well for these.

GAPBS is distributed under its own licence. See the upstream repository.
