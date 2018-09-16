#_*_ coding:utf-8_*_
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt
from gen_config import *
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--seq', default='', help='input seq')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
    print len(img_list)
    dpi = 80.0
    interval = float(1 / 24)
    image = Image.open(img_list[0]).convert('RGB')
    figsize = (image.size[0] / dpi, image.size[1] / dpi)
    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()  # 关闭坐标轴
    fig.add_axes(ax)
    im = ax.imshow(image, aspect='1')
    gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                            linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
    ax.add_patch(gt_rect)
    plt.pause(.1)
    plt.draw()
    for i in range(1, len(img_list)):
        image = Image.open(img_list[i]).convert('RGB')
        im.set_data(image)
        print gt[i]
        gt_rect.set_xy(gt[i, :2])
        gt_rect.set_width(gt[i, 2])
        gt_rect.set_height(gt[i, 3])
        plt.pause(interval=.1)
        plt.draw()
