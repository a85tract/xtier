// xtier_intent.h
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _XTIER_INTENT_H
#define _XTIER_INTENT_H

#include <linux/types.h>

enum xtier_action {
    XTIER_PROMOTE = 1,
    XTIER_DEMOTE  = 2,
};

struct xtier_intent {
    int           pid;
    unsigned long start;
    unsigned long end;
    unsigned int  target_node;
    unsigned int  action;
    int           score;
};

#endif /* _XTIER_INTENT_H */
