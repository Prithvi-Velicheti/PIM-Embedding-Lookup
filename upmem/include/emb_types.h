#include <stdint.h>
#include <stdbool.h>

#define MAX_ENC_BUFFER_SIZE MEGABYTE(MAX_ENC_BUFFER_MB)
#define MAX_CAPACITY MEGABYTE(14) //Must be a multiply of 2
#define DPUS_PER_RANK 64
#define AVAILABLE_RANKS 20
#define MAX_NR_BUFFERS 65

struct buffer_meta {
    uint32_t col_id;
    uint32_t table_id;
} __attribute__((packed));

struct embedding_table {
    uint32_t rank_id;
    struct dpu_set_t *rank;
    uint64_t nr_rows;
};

struct query_len {
    uint32_t indices_len;
    uint32_t nr_batches;
}__attribute__((packed));

struct callback_input{
    float** final_results;
    uint32_t nr_batches;
    int32_t*** tmp_results;
}; 

struct metadata {
	uint32_t indices_len ; 
	uint32_t offsets_len ; 
	uint32_t embedding_data_len ; 
}; 

typedef struct get_block_t {
    /** The get_block function */
    get_block_func_t f;
    /** User arguments for the get_block function */
    void *args;
} get_block_t;

struct sg_block_info {
    /** Starting address of the block */
    uint8_t *addr;
    /** Number of bytes to transfer for this block */
    uint32_t length;
};

typedef bool (*get_block_func_t)(struct sg_block_info *out, uint32_t dpu_index, uint32_t block_index, void *args);













