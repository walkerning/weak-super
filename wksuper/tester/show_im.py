# -*- coding: utf-8 -*-
"""
显示图片钩子
"""
import matplotlib.pyplot as plt

from .hook import Hook

class ShowImageHook(Hook):
    TYPE = "ShowImage"

    def post_test(self, *args):
        im_ind, rois_score = args
        if not rois_score:
            print u"检测结果: 图片 {} 没有任何物体".format(im_ind)
        else:
            print u"检测结果: 图片 {} 有 {} 个检测结果".format(im_ind, len(rois_score))
            self.visualize(im_ind, rois_score)

    def visualize(self, im_ind, rois_score):
        im = self.tester.dataset.get_image_by_id(im_ind)
        fig, ax = plt.subplots(figsize=(12, 12))
        # draw the original image
        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect="equal")

        for bbox, (det_name, det_score) in rois_score.iteritems():
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{} {}'.format(det_name, det_score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections'.format(im_ind)),
                     fontsize=14)
        print "image %s handled, %d boxes ploted" % (im_ind, len(rois_score))
        plt.axis("off")
        plt.tight_layout()
        plt.draw()
        plt.show() # will block by default
