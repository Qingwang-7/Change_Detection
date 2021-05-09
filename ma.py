# -*- coding:utf-8 -*-
from PIL import Image
import os


def cut(image_path, save_path, vx, vy):
    count = 0

    im_name = os.listdir(image_path)
    paths = []
    for name in im_name:
        path = os.path.join(image_path, name)
        paths += [path]
    for i, path in enumerate(paths):

        name = (path.split('/')[-1]).split('.')[0]
        name2 = save_path + name + '_'

        im = Image.open(path)
        w = im.size[0]
        h = im.size[1]

        # print(w, h)
        # 偏移量
        dx = 300
        dy = 300
        n = 1

        # 左上角切割
        x1 = 0
        y1 = 0
        x2 = vx
        y2 = vy
        # 纵向
        while x2 <= h:
            while y2 <= w:
                name3 = name2 + '%06d' % (n) + ".png"
                # print(n, x1, y1, x2, y2)
                im2 = im.crop((y1, x1, y2, x2))
                im2.save(name3)
                y1 = y1 + dy
                y2 = y1 + vy
                n = n + 1
            if y2 >= w:
                name3 = name2 + '%06d' % (n) + ".png"
                # print(n, x1, y1, x2, y2)
                y1 = w - vy
                y2 = w
                im2 = im.crop((y1, x1, y2, x2))
                im2.save(name3)
                # print n, x1, y1, x2, y2
                n = n + 1
            x1 = x1 + dx
            x2 = x1 + vx
            y1 = 0
            y2 = vy
        x1 = h - vx
        x2 = h
        y1 = 0
        y2 = vy
        while y2 <= w:
            name3 = name2 + '%06d' % (n) + ".png"
            # print(n, x1, y1, x2, y2)
            im2 = im.crop((y1, x1, y2, x2))
            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1
        if y2 >= w:
            name3 = name2 + '%06d' % (n) + ".png"
            # print(n, x1, y1, x2, y2)
            y1 = w - vy
            y2 = w
            im2 = im.crop((y1, x1, y2, x2))
            im2.save(name3)
            n = n + 1

        print(i + 1, '/', len(paths))
        count += n
    return count


if __name__ == "__main__":
    # 'F:/DL_Code/STANet-master/path-to-LEVIR-CD-test/label/'
    image_path = 'F:/DL_Code/data/data_2/SZTAKI/Szada/SZTAKI_train/label/'
    save_path = 'F:/DL_Code/data/data_2/SZTAKI/Szada/SZTAKI_160_train/label/'

    # 切割图片的面积 vx,vy
    # 大
    res = cut(image_path, save_path, 320, 479)

    # 中
    # res = cut(id,120,120)

    # 小
    # res = cut(id,80,80)

    print('all sub image:', res)