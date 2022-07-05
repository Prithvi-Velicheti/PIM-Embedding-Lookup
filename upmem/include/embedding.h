#include "common.h"
#include "host/include/host.h"
#include "emb_types.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>


#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @brief TBC
 * @param TBC
 */
typedef struct dpu_runtime_totals {
    double execution_time_prepare;
    double execution_time_populate_copy_in;
    double execution_time_copy_in;
    double execution_time_copy_out;
    double execution_time_aggregate_result;
    double execution_time_launch;
} dpu_runtime_totals;

/**
 * @brief TBC
 * @param TBC
 */
typedef struct dpu_timespec {
    long tv_nsec;
    long tv_sec;
} dpu_timespec;

/**
 * @brief TBC
 * @param TBC
 */
typedef struct dpu_runtime_interval {
    dpu_timespec start;
    dpu_timespec stop;
} dpu_runtime_interval;

/**
 * @brief TBC
 * @param TBC
 */
typedef struct dpu_runtime_group {
    unsigned int in_use;
    unsigned int length;
    dpu_runtime_interval *intervals;
} dpu_runtime_group;

// static void
// copy_interval(dpu_runtime_interval *interval, struct timespec *const start,
//               struct timespec *const end);

void alloc_dpus(uint64_t nr_dpus);

void
populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data,
              dpu_runtime_totals *runtime);


dpu_error_t
post_process(struct dpu_set_t dpu_rank, uint32_t rank_id, void *arg);

int32_t *
lookup(uint32_t **indices, uint32_t **offsets, uint32_t *indices_len, uint32_t *nr_batches,
       float **final_results);