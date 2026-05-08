

from pathlib import Path
def main():
    # 将/data2/code/Semi-supervised-learning/data/cbcnet/geetest-four-ch-little,下的标注图像("_" 前为标签)，生成2列，一列为图片路径，另一列为图片标签
    root_dir='/data2/code/Semi-supervised-learning/data/cbcnet/test_data'
    pic_paths=Path(root_dir)
    pic_paths=pic_paths.glob('**/*.jpg')
    pic_list=[str(pic_path) for pic_path in pic_paths]
    for pic_path_ in pic_list:
        pic_path=str(pic_path_)
        pic_name=pic_path.split('/')[-1]
        pic_label=pic_name.split('_')[0]
        print(pic_label)

if __name__ == '__main__':
    main()