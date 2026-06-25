import cv2
import numpy as np
from pathlib import Path
def vis_yolo_data(data_path):
    for img_path in data_path.glob("**/*.jpg"):
        img=cv2.imread(str(img_path))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        txt_path=img_path.with_suffix(".txt")
        with open(txt_path) as f:
            lines=f.readlines()
        for l in lines:
            l=l.strip().split(" ")
            x1,y1,x2,y2=map(float, l[1:5])
            img=cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # data_dir="/media/mpf/fssd21/damagou_error/yidun_ch_click20260606_1500/"
    data_dir="/media/mpf/fssd21/damagou_error/test/"
    data_path=Path(data_dir)
    jpeg_paths=data_path.glob("**/*.jpg")
    for jpeg_path in jpeg_paths:
        print(str(jpeg_path))

    #data_dir 中保存了jpg图像和对应的目标检测框的txt文件，可视化出来以便检查数据是否标注正确
    # vis_yolo_data(Path(data_dir))