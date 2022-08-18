#RM1

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=32
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=80
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=400000
NR_EMBEDDING=8
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0


[PERF] CPU CLOCK RAW time [ms]: 1415.37
[PERF] CPUTIME time [ms]: 8330.78
[PERF] CPU PROCESS ratio: 588.59
[PERF] CPU CLOCK RAW time [ms]: 1141.85
[PERF] CPUTIME time [ms]: 6244.13
[PERF] CPU PROCESS ratio: 546.84
[PERF] CPU CLOCK RAW time [ms]: 887.73
[PERF] CPUTIME time [ms]: 4968.43
[PERF] CPU PROCESS ratio: 559.68
[PERF] CPU CLOCK RAW time [ms]: 730.26
[PERF] CPUTIME time [ms]: 3937.23
[PERF] CPU PROCESS ratio: 539.15
[PERF] CPU CLOCK RAW time [ms]: 651.48
[PERF] CPUTIME time [ms]: 3339.46
[PERF] CPU PROCESS ratio: 512.59
[PERF] CPU CLOCK RAW time [ms]: 560.93
[PERF] CPUTIME time [ms]: 2811.62
[PERF] CPU PROCESS ratio: 501.24
[PERF] CPU CLOCK RAW time [ms]: 532.91
[PERF] CPUTIME time [ms]: 2670.19
[PERF] CPU PROCESS ratio: 501.05
[PERF] CPU CLOCK RAW time [ms]: 514.86
[PERF] CPUTIME time [ms]: 2523.09
[PERF] CPU PROCESS ratio: 490.05
[PERF] CPU CLOCK RAW time [ms]: 547.81
[PERF] CPUTIME time [ms]: 2607.10
[PERF] CPU PROCESS ratio: 475.91
[PERF] CPU CLOCK RAW time [ms]: 559.90
[PERF] CPUTIME time [ms]: 2776.51
[PERF] CPU PROCESS ratio: 495.90
[PERF] CPU CLOCK RAW time [ms]: 534.22
[PERF] CPUTIME time [ms]: 2536.78
[PERF] CPU PROCESS ratio: 474.86
[PERF] CPU CLOCK RAW time [ms]: 538.11
[PERF] CPUTIME time [ms]: 2593.65
[PERF] CPU PROCESS ratio: 481.99
[PERF] CPU CLOCK RAW time [ms]: 500.13
[PERF] CPUTIME time [ms]: 2480.87
[PERF] CPU PROCESS ratio: 496.04
[PERF] CPU CLOCK RAW time [ms]: 541.61
[PERF] CPUTIME time [ms]: 2744.76
[PERF] CPU PROCESS ratio: 506.78
[PERF] CPU CLOCK RAW time [ms]: 497.94
[PERF] CPUTIME time [ms]: 2460.10
[PERF] CPU PROCESS ratio: 494.06
[PERF] CPU CLOCK RAW time [ms]: 556.38
[PERF] CPUTIME time [ms]: 2695.28
[PERF] CPU PROCESS ratio: 484.43
[PERF] CPU CLOCK RAW time [ms]: 539.59
[PERF] CPUTIME time [ms]: 2719.95
[PERF] CPU PROCESS ratio: 504.08
[PERF] CPU CLOCK RAW time [ms]: 491.21
[PERF] CPUTIME time [ms]: 2467.48
[PERF] CPU PROCESS ratio: 502.33
[PERF] CPU CLOCK RAW time [ms]: 595.94
[PERF] CPUTIME time [ms]: 2515.64
[PERF] CPU PROCESS ratio: 422.13
[PERF] CPU CLOCK RAW time [ms]: 494.96
[PERF] CPUTIME time [ms]: 2415.80
[PERF] CPU PROCESS ratio: 488.08


dpu [ms]: 7244.577750, cpu [ms] 641.659900, dpu acceleration 0.088571
 DPU PRATIO 104.293458, CPU PRATIO 503.289621, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

#nombre max de RM1/server 1->96 DPU
N = 2304 * 8 /9
on prend N = 64 -> 2048 emb  : 2048 -> BUG de taille

test avec N= 63 -> 2016 emb
test avec N= 61~62 emb -> 2000 emb


[PERF] CPU CLOCK RAW time [ms]: 295819.52
[PERF] CPUTIME time [ms]: 4568571.39
[PERF] CPU PROCESS ratio: 1544.38

first iteration -> 16

dpu [ms]: 17521.255700, cpu [ms] 131609.255200, dpu acceleration 7.511405
 DPU PRATIO 1008.385525, CPU PRATIO 3005.624527, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)




#RM2

MAX_NR_DPUS=10000
MAX_NR_EMBEDDING=2000
MAX_INDEX_PER_BATCH=160
MAX_NR_BATCHES=60
NR_COLS=64
NR_BATCHES=60
NR_RUN=1
INDEX_PER_BATCH=120
MAX_INDEX_PER_BATCH_RAND=160
NR_ROWS=500000
NR_EMBEDDING=32
NR_TASKLETS=16
CHECK_RESULTS=1
RAND_INPUT_SIZE=0

[PERF] CPU CLOCK RAW time [ms]: 590271.87
[PERF] CPUTIME time [ms]: 5254732.80
[PERF] CPU PROCESS ratio: 890.22
[PERF] CPU CLOCK RAW time [ms]: 142576.42
[PERF] CPUTIME time [ms]: 3415347.71
[PERF] CPU PROCESS ratio: 2395.45
[PERF] CPU CLOCK RAW time [ms]: 113854.77
[PERF] CPUTIME time [ms]: 3378877.18
[PERF] CPU PROCESS ratio: 2967.71
[PERF] CPU CLOCK RAW time [ms]: 111953.92
[PERF] CPUTIME time [ms]: 3390876.42
[PERF] CPU PROCESS ratio: 3028.81
[PERF] CPU CLOCK RAW time [ms]: 112745.96
[PERF] CPUTIME time [ms]: 3398640.38
[PERF] CPU PROCESS ratio: 3014.42
[PERF] CPU CLOCK RAW time [ms]: 111383.07
[PERF] CPUTIME time [ms]: 3410856.70
[PERF] CPU PROCESS ratio: 3062.28
[PERF] CPU CLOCK RAW time [ms]: 110246.59
[PERF] CPUTIME time [ms]: 3375586.82
[PERF] CPU PROCESS ratio: 3061.85
[PERF] CPU CLOCK RAW time [ms]: 214738.70
[PERF] CPUTIME time [ms]: 3423910.66
[PERF] CPU PROCESS ratio: 1594.45
[PERF] CPU CLOCK RAW time [ms]: 109773.90
[PERF] CPUTIME time [ms]: 3332821.50
[PERF] CPU PROCESS ratio: 3036.08
[PERF] CPU CLOCK RAW time [ms]: 108513.61
[PERF] CPUTIME time [ms]: 3318558.72
[PERF] CPU PROCESS ratio: 3058.20
[PERF] CPU CLOCK RAW time [ms]: 109251.18
[PERF] CPUTIME time [ms]: 3357027.07
[PERF] CPU PROCESS ratio: 3072.76
[PERF] CPU CLOCK RAW time [ms]: 109490.18
[PERF] CPUTIME time [ms]: 3362308.10
[PERF] CPU PROCESS ratio: 3070.88
[PERF] CPU CLOCK RAW time [ms]: 109558.07
[PERF] CPUTIME time [ms]: 3373818.88
[PERF] CPU PROCESS ratio: 3079.48
[PERF] CPU CLOCK RAW time [ms]: 109036.23
[PERF] CPUTIME time [ms]: 3350740.22
[PERF] CPU PROCESS ratio: 3073.05
[PERF] CPU CLOCK RAW time [ms]: 109264.14
[PERF] CPUTIME time [ms]: 3353164.54
[PERF] CPU PROCESS ratio: 3068.86
[PERF] CPU CLOCK RAW time [ms]: 109701.05
[PERF] CPUTIME time [ms]: 3364119.04
[PERF] CPU PROCESS ratio: 3066.62
[PERF] CPU CLOCK RAW time [ms]: 112337.65
[PERF] CPUTIME time [ms]: 3375103.49
[PERF] CPU PROCESS ratio: 3004.43
[PERF] CPU CLOCK RAW time [ms]: 110663.14
[PERF] CPUTIME time [ms]: 3392515.07
[PERF] CPU PROCESS ratio: 3065.62
[PERF] CPU CLOCK RAW time [ms]: 111403.22
[PERF] CPUTIME time [ms]: 3385780.74
[PERF] CPU PROCESS ratio: 3039.21
[PERF] CPU CLOCK RAW time [ms]: 110388.59
[PERF] CPUTIME time [ms]: 3371287.81
[PERF] CPU PROCESS ratio: 3054.02

dpu [ms]: 10559.500400, cpu [ms] 10968.014500, dpu acceleration 1.038687
 DPU PRATIO 121.276600, CPU PRATIO 1851.410540, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)




#nombre max de RM2/server 1->96 DPU
N = 2304 * 32 /96
on prend N =24-1  -> 736 emb




alloc FIFO [build_synthetic_input_data->inference], DEPTH(2)
map embeddings on DPUs
min nr cols per dpu 2
nr cols per dpus 28, dpu part col 8
MRAM_SIZE 67108864 DPU_EMB_DATA_SIZE_BYTE 58720256 nr cols per dpus 28
nr_dpus 2208
nr cols per dpu 28
alloc dpus 2208
generate synthetic tables
populate mram with embedding synthetic tables
start xfer 0 part dpus with size 16000000 nr cols 8
start inference
max nr embedding 2000


NR BATCHES = 60
dpu [ms]: 19269.069800, cpu [ms] 141357.613200, dpu acceleration 7.335985
 DPU PRATIO 1007.189880, CPU PRATIO 2835.220536, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


batch size = 32 

dpu [ms]: 11230.810750, cpu [ms] 106665.612800, dpu acceleration 9.497588
 DPU PRATIO 922.110314, CPU PRATIO 2779.491364, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

batch size = 16


dpu [ms]: 7244.730700, cpu [ms] 40929.359600, dpu acceleration 5.649535
 DPU PRATIO 843.557799, CPU PRATIO 2593.819623, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)


batch size = 8 

dpu [ms]: 5795.315450, cpu [ms] 53081.817550, dpu acceleration 9.159435
 DPU PRATIO 768.407324, CPU PRATIO 2339.628122, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)

batch size = 1

dpu [ms]: 4678.842500, cpu [ms] 9889.886500, dpu acceleration 2.113746
 DPU PRATIO 671.029614, CPU PRATIO 990.274037, DPU OK ? 1 
free FIFO [build_synthetic_input_data->inference], DEPTH(2)
