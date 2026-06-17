
if __name__ == '__main__':
    train_base_data_txt1="/data2/fssd2/damagou_error/train.txt"

    lines=[]
    with open(train_base_data_txt1) as f:
        lines = f.readlines()

    from semilearn.cbc_config import keys
    lines1=[]
    for l in lines:
        img_path, label = l.strip().split(" ")
        if label.strip() in keys.alphabet:
            lines1.append(l.strip())
    with open("/data2/fssd2/damagou_error/train_tmp.txt", "w") as f:
        f.write("\n".join(lines1))