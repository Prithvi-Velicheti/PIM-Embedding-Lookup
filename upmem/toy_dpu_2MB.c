// To build the code: dpu-upmem-dpurte-clang -o toy_dpu_2MB toy_dpu_2MB.c
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

__mram_noinit int32_t data[2097152];

__mram_noinit uint64_t row_size_input;
__mram_noinit uint64_t col_size_input;
__mram_noinit uint64_t first_index_input;
__mram_noinit uint64_t last_index_input;

int main() { 
    __dma_aligned int32_t read_buf[2097152];

    for (int i=0; i<2097152; i++){
        read_buf[i]=data[i];
        printf("%dth is %d\n", i, data[i]);
    }

    //write the contents of the write_buf in MRAM, replacing the contents of read_buf
    mram_write((const int32_t*)read_buf, DPU_MRAM_HEAP_POINTER, 2097152*sizeof(int32_t));
    
    return 0;
}