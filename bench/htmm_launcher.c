/* SPDX-License-Identifier: Apache-2.0 */
/* htmm_launcher: fork+exec a workload, htmm_start its PID, htmm_end on exit.
 * Usage: htmm_launcher <node> -- <cmd> [args...]
 *
 * Memtis only tracks a process once htmm_start has been called for its PID;
 * writing memory.htmm_enabled is not enough. Needs a kernel built with
 * kernel/build.sh --stack=all. Build with: cc -O2 -o htmm_launcher htmm_launcher.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <errno.h>
#include <string.h>

#define __NR_htmm_start 463
#define __NR_htmm_end   464

int main(int argc, char **argv) {
    if (argc < 4 || strcmp(argv[2], "--") != 0) {
        fprintf(stderr, "usage: %s <node> -- <cmd> [args...]\n", argv[0]);
        return 1;
    }
    int node = atoi(argv[1]);
    char **child_argv = &argv[3];

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        /* Sleep briefly so parent can htmm_start before workload runs hot */
        usleep(100000);
        execvp(child_argv[0], child_argv);
        perror("execvp");
        _exit(127);
    }

    /* Parent */
    long ret = syscall(__NR_htmm_start, pid, node);
    fprintf(stderr, "[htmm_launcher] htmm_start(pid=%d, node=%d) -> %ld errno=%d\n",
            pid, node, ret, errno);

    int status = 0;
    waitpid(pid, &status, 0);

    long endret = syscall(__NR_htmm_end, pid);
    fprintf(stderr, "[htmm_launcher] htmm_end(pid=%d) -> %ld errno=%d\n",
            pid, endret, errno);

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128 + WTERMSIG(status);
}
