#1-重新开始训练，2-继续，3-微调

import math

from semilearn.cbc_config import keys

model=1
#是否样本增强
is_random_enhance_train=False
is_random_enhance_val=False
input_h=32
input_w=32
nclass=len(keys.alphabet)
lr=0.0001
epoches=5000
train_bs=16
val_bs=16
show_interval=1
save_interval_epoches=1
dropout_ratio=0.5
# kernel_size=(1,math.ceil(config.input_w/8/timestamp))   real_timestamp=math.ceil(config.input_w/8.0/kernel_size[1])
timestamp=1 #math.ceil(input_w/8.0)
alphabet=keys.alphabet
decode_raw=False
gpu_no = 0
device_ids = [0,1,2]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
val_nw = 0  #dataloader 中的 num_workers 参数
train_nw = 0
#time optimize params
check_points=False
fp16 = False
auto_cast_device = 'cuda'