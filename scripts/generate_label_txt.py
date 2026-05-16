from pathlib import Path
import random
# nohup python -u train.py --c config/usb_cv/softmatch/softmatch_cbcnet1.yaml 2>&1 > logs/yidun1.log &
if __name__ == "__main__":
    # root_dir = Path("/data2/fssd2/damagou_error/fist_500_ul_pics")
    # lines=[]
    # for img_path in root_dir.glob("**/*.jpg"):
    #     label=img_path.stem.split("_")[0]
    #     if len(label)!=1:
    #         continue
    #     lines.append(f"{str(img_path)} {label}")
    # random.shuffle(lines)
    # with open("/data2/fssd2/damagou_error/train_tmp.txt", "w") as f:
    #     f.write("\n".join(lines))
    with open("/data2/fssd2/damagou_error/train.txt") as f:
        lines = f.readlines()

    from semilearn.cbc_config import keys
    lines1=[]
    for l in lines:
        img_path, label = l.strip().split(" ")
        if label.strip() in keys.alphabet:
            lines1.append(l.strip())
    with open("/data2/fssd2/damagou_error/train_tmp1.txt", "w") as f:
        f.write("\n".join(lines1))