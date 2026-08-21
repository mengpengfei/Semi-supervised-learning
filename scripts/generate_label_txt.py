from pathlib import Path
import random
import os
import math

def dirpath(root_dir,lpath,test_ratio):
    train_filelist=[]
    test_filelist=[]

    for path in lpath:
        lfilelist=[]
        sub_dir=os.path.join(root_dir, path)
        for root, dirs, files in os.walk(sub_dir):
            for file in files:
                if file.find(" ")>=0: #or not file.endswith("bmp"):
                    continue
                else:
                    lfilelist.append(os.path.join(root, file))
        random.shuffle(lfilelist)
        offset=math.floor(len(lfilelist)*test_ratio)
        train_filelist_tmp=lfilelist[offset:]
        test_filelist_tmp=lfilelist[:offset]
        train_filelist=train_filelist_tmp+train_filelist
        test_filelist=test_filelist_tmp+test_filelist

    random.shuffle(train_filelist)
    random.shuffle(test_filelist)
    return test_filelist,train_filelist


# nohup python -u train.py --c config/usb_cv/softmatch/softmatch_cbcnet1.yaml 2>&1 > logs/yidun1.log &
# nohup python -u train.py --c config/classic_cv_imb/softmatch_abc/softmatch_cbcnet_abc.yaml 2>&1 > logs/yidunabc20260624.log &
if __name__ == "__main__":
    base_labeled_data_root_dir='/data2/fssd2/jiyan/jiyan-big-origin-labeled-used/'
    unlabled_data_root_dir='/data2/fssd2/damagou_error/ch/unlabeled/'
    txt_dst_path="/data2/fssd2/damagou_error/"

    base_labled_str=[r"ch-error-2021-08-28"
        ,r"ch-error-2021-09-02"
        ,r"ch-error-2021-09-09"
              # ,r"ch-error-2021-09-29"
        ,r"ch-error-2021-10-11"
              #         ,r"ch-error-2021-11-05"
        ,r"ch-error-2021-11-05"
              #         ,r"ch-error-2021-11-12"
        ,r"ch-error-2021-11-12"
              #         ,r"ch-error-2021-11-15"
        ,r"ch-error-2021-11-15"
              #         ,r"ch-error-2021-11-16"
        ,r"ch-error-2021-11-16"
              #         ,r"ch-error-2021-11-17"
        ,r"ch-error-2021-11-17"
              #         ,r"ch-error-2022-01-26"
        ,r"ch-error-2022-01-26"
        ,r"ch-error-2022-02-08"
        ,r"ch-error-2022-02-19"
              #         ,r"ch-error-2022-02-19"
        ,r"ch-error-2022-02-22"
        ,r"ch-error-2022-04-20"
        ,r"ch-error-2022-04-21"
        ,r"ch-error-2022-04-21_1"
        ,r"ch-error-2022-04-22"
        ,r"ch-error-2022-04-22_1"
        ,r"ch-error-2022-11-20_1"

        ,r"geetest4-ch-big-2023-11-03"
        ,r"ch-error-2024-09-03"
        ,r"ch-error-2024-09-03"
        ,r"ch-error-2024-09-24"
        ,r"ch-error-2024-09-24"
        ,r"ch-error-2025-03-31"
        ,r"ch-error-2025-03-31_1"


        ,r"geetest4-ch-big-2022-04-21"
        ,r"geetest4-ch-big-2022-04-21"

        ,r"geetest4-ch-2021-11-28"
        ,r"geetest-four-ch-little"
        ,r"error-back-2021-02-16"
        ,r"error-back-2021-03-30"
        ,r"jiyanhanzi_36w/old_renamed"
        ,r"jiyanhanzi_36w/renamed"
        ,r"jiyan-singile/jiyan-crop"
        ,r"jiyan-singile/jiyan-crop1"

        ,r"fist_500_ul_pics"
        ,r"fist_500_ul_pics"
        ,r"txt_pred_20260516"
        ,r"txt_pred_20260516"
        ,r"yidun_ch_click20260606_cropped_5000"
        ,r"yidun_ch_click20260606_cropped_5000"
        ,r"geetest4-ch-big-20260804_cropped"
        ,r"geetest4-ch-big-20260804_cropped"

              ]

    base_unlabled_str=[
        r"yidun_ch_click20260322_label_croped",
        r"yidun_ch_click20260606_cropped_0",
        r"geest4-ch-20260804_cropped"
    ]


    base_labeld_test_filelist,base_labeld_train_filelist = dirpath(base_labeled_data_root_dir, base_labled_str,0.001)
    unlabeld_test_filelist,unlabeld_train_filelist = dirpath(unlabled_data_root_dir, base_unlabled_str,0)

    labeled_lines=[]
    split_char='_'
    for img_path in base_labeld_train_filelist:
        label=Path(img_path).stem.split(split_char)[0]
        if len(label.strip())>1:
            continue
        labeled_lines.append(f"{str(img_path)} {label}\n")

    val_labeled_lines=[]
    split_char='_'
    for img_path in base_labeld_test_filelist:
        label=Path(img_path).stem.split(split_char)[0]
        if len(label.strip())>1:
            continue
        val_labeled_lines.append(f"{str(img_path)} {label}\n")

    unlabeled_lines=[]
    for img_path in unlabeld_train_filelist:
        # label=Path(img_path).stem.split(split_char)[0]
        unlabeled_lines.append(f"{str(img_path)}\n")

    random.shuffle(labeled_lines)
    random.shuffle(val_labeled_lines)
    random.shuffle(unlabeled_lines)

    labeled_txt_path=f"{txt_dst_path}/train.txt"
    labeled_val_txt_path=f"{txt_dst_path}/val.txt"
    unlabeled_txt_path=f"{txt_dst_path}/train-ulb.txt"

    if os.path.exists(labeled_txt_path):
        os.remove(labeled_txt_path)

    if os.path.exists(labeled_val_txt_path):
        os.remove(labeled_val_txt_path)

    if os.path.exists(unlabeled_txt_path):
        os.remove(unlabeled_txt_path)

    with open(labeled_txt_path,'w+',encoding='utf-8') as fi:
        for line in labeled_lines:
            print(line)
            fi.write(str(line))

    with open(labeled_val_txt_path,'w+',encoding='utf-8') as fi:
        for line in val_labeled_lines:
            print(line)
            fi.write(str(line))

    with open(unlabeled_txt_path,'w+',encoding='utf-8') as fi:
        for line in unlabeled_lines:
            print(line)
            fi.write(str(line))

