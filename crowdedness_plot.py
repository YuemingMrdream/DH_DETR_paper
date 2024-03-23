import json
import os
import numpy as np
import sys

from matplotlib import pyplot as plt

PERSON_CLASSES = ['background', 'person']

class Image(object):
    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def load(self, record, body_key, head_key, class_names, gtflag):
        """
        :meth: read the object from a dict
        """
        if "ID" in record and self.ID is None:
            self.ID = record['ID']
        if "width" in record and self._width is None:
            self._width = record["width"]
        if "height" in record and self._height is None:
            self._height = record["height"]
        if gtflag:
            self._gtNum = len(record["gtboxes"])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes', class_names)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array(
                    [body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                     for i in range(len(body_bbox))]
                )
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack(
                    (body_bbox[:, :-1], head_bbox[:, :-1], gt_tag.reshape(-1, 1))
                )
            else:
                raise Exception('Unknown evaluation mode!')
        if not gtflag:
            self._dtNum = len(record["dtboxes"])
            if self.eval_mode == 0:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key, 'score')
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None or self.gtboxes is None:
            return list()

        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True)) #按照预测分数排序
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            #每一个dtbox与所有gtbox计算IOU
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] > 0], True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] <= 0], False)
            overlap_gt_iou = self.box_overlap_opr(gtboxes[gtboxes[:, -1] > 0], gtboxes[gtboxes[:, -1] > 0], True)
            overlap_gt_iou[np.diag_indices_from(overlap_gt_iou)] *= 0
            ign = np.any(overlap_ioa > thres, 1)
            pos = np.any(overlap_iou > thres, 1)
        else:
            return list()

        cdlist = [[0 for col in range(2)] for row in range(gtboxes[gtboxes[:, -1] > 0].shape[0])]

        for i in range(gtboxes[gtboxes[:, -1] > 0].shape[0]):
            cdpos = np.argmax(overlap_gt_iou[i])
            crowdedness = overlap_gt_iou[i, cdpos]
            cdlist[i][0] = int(crowdedness*10)/10

        for i, dt in enumerate(dtboxes):
            maxpos = np.argmax(overlap_iou[i]) #找当前dt对应gt的最大iou值对应的索引
            if overlap_iou[i, maxpos] > thres: #如果当前dt对应的最大iou值》阈值
                if dt[-1] > 0.7: # 置信度
                    overlap_iou[:, maxpos] = 0 #列赋值为0，绑定当前dt
#                scorelist.append((dt, 1, self.ID, pos[i]))
                #scorelist.append((int(dt[-1]*10)/10, int(crowdedness*10)/10, 1)) #标记为1
                    cdlist[maxpos][1] = 1 #标记为1
            elif not ign[i]:
                #scorelist.append((dt, 0, self.ID, pos[i]))
                # scorelist.append((int(dt[-1]*10)/10, int(crowdedness*10)/10, 0))
                cdlist[maxpos][1] = 0 #标记为0

        return cdlist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:, None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:, None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def box_overlap_opr(self, dboxes: np.ndarray, gboxes: np.ndarray, if_iou) -> np.ndarray:
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

        iw = (np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2])
              - np.maximum(dtboxes[:, :, 0], gtboxes[:, :, 0]))
        ih = (np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3])
              - np.maximum(dtboxes[:, :, 1], gtboxes[:, :, 1]))
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:, :, 2] - dtboxes[:, :, 0]) * (dtboxes[:, :, 3] - dtboxes[:, :, 1])
        if if_iou:
            gtarea = (gtboxes[:, :, 2] - gtboxes[:, :, 0]) * (gtboxes[:, :, 3] - gtboxes[:, :, 1])
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dtboxes.shape[-1] >= 4
        assert self.gtboxes.shape[-1] >= 4
        # assert self._width is not None and self._height is not None
        # if self.eval_mode == 2:
        #     self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4], self._height, self._width)
        #     self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4], self._height, self._width)
        #     self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8], self._height, self._width)
        #     self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8], self._height, self._width)
        # else:
        #     self.dtboxes = _clip_boundary(self.dtboxes, self._height, self._width)
        #     self.gtboxes = _clip_boundary(self.gtboxes, self._height, self._width)

    def load_gt_boxes(self, dict_input, key_name, class_names):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = 1
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            # head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            body_bbox.append((*rb['fbox'], body_tag))
        # head_bbox = np.array(head_bbox)
        # head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    def load_det_boxes(self, dict_input, key_name, key_box, key_score=None, key_tag=None):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack(
                    [
                        np.hstack(
                            (rb[key_box], rb[key_score], rb[key_tag])
                        ) for rb in dict_input[key_name]
                    ]
                )
            else:
                bboxes = np.array([(*rb[key_box], rb[key_score]) for rb in dict_input[key_name]])
        else:
            if key_tag:
                bboxes = np.vstack(
                    [np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]]
                )
            else:
                bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist

class Database(object):
    def __init__(self, gtpath=None, dtpath=None, body_key=None, head_key=None, mode=0):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.eval_mode = mode
        self.loadData(gtpath, body_key, head_key, if_gt=True)
        self.loadData(dtpath, body_key, head_key, if_gt=False)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        with open(fpath, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        if if_gt:
            for record in records:
                    self.images[record["ID"]] = Image(self.eval_mode)
                    self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
        else:
            for record in records:
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False)
                self.images[record["ID"]].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            # scorelist.append((dt, 0, self.ID, pos[i]))
            scorelist.extend(result)

        # In the descending sort of dtbox score.
        #scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2", fppiX=None, fppiY=None):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst) - 1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        if fppiX is None or fppiY is None:
            fppiX, fppiY = list(), list()
            for i, item in enumerate(self.scorelist):
                if item[1] == 1:
                    tp += 1.0
                elif item[1] == 0:
                    fp += 1.0

                fn = (self._gtNum - self._ignNum) - tp
                recall = tp / (tp + fn)
                missrate = 1.0 - recall
                fppi = fp / self._imageNum
                fppiX.append(fppi)
                fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i - 1] + precision[i]) / 2
                delta_w = recall[i] - recall[i - 1]
                area += delta_w * delta_h
            return area

        tp, fp, dp = 0.0, 0.0, 0.0
        rpX, rpY = list(), list()
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        dpn = []
        recalln = []
        thr = []
        mr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
                dp += item[-1]
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            dpn.append(dp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / total_images)
            mr.append(1 - recall)

        AP = _calculate_map(rpX, rpY)
        return AP, recall, (rpX, rpY, thr, fpn, dpn, recalln, fppi, mr)

def caculate_iou(gt_path, dt_path, thres, target_key="box", mode=0):
    database = Database(gt_path, dt_path, target_key, None, mode)
    database.compare(thres) #gt和dt的匹配
    tps = []
    fps = []

    for crowdedness in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        tp = database.scorelist.count([crowdedness,1])
        fp = database.scorelist.count([crowdedness,0])
        print('crowdedness: {}, tp: {}'.format(crowdedness, tp))
        print('crowdedness: {}, fp: {}'.format(crowdedness, fp))
        tps.append(tp)
        fps.append(fp)
    # tps = tps[2:]
    # fps = fps[2:]
    return tps, fps

if __name__ == "__main__":
    # gt_path = './annotation_val.odgt'
    # dt_path1 = './epoch-55.human' #good
    # dt_path2 = './epoch-55(1).human' #baseline
    gt_path = './annotation_val.odgt'
    dt_path2 = './DH_head/eval_dump/two-stage-deformable/epoch-51.human' #baseline
    dt_path1 = './DH_head/eval_dump/two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500/epoch-51.human'


    for thres in [0.5,0.6,0.7,0.8,0.9]:

        tps1,fps1 = caculate_iou(gt_path, dt_path1, thres)
        tps2,fps2 = caculate_iou(gt_path, dt_path2, thres)

        # tps2 = [4233, 4859, 5208, 5905, 6665, 7450, 8738, 45293]
        # fps2 = [37292, 11626, 4628, 2396, 1445, 922, 702, 1717]
        x_data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        length = len(tps1)
        x = np.arange(length) # 横坐标范围

        plt.figure()
        total_width, n = 1, 2   # 柱状图总宽度，有几组数据
        width = total_width / n   # 单个柱状图的宽度
        x1 = x - width / 2   # 第一组数据柱状图横坐标起始位置
        x2 = x1 + width   # 第二组数据柱状图横坐标起始位置

        # plt.figure()
        f, axs = plt.subplots(2, 1, sharex=True)

        tps = list(map(lambda x: (x[0] - x[1])/x[0], zip(tps1, tps2)))
        fps = list(map(lambda x: (x[0] - x[1])/x[0], zip(fps1, fps2)))

        axs[0].bar(x1, tps, width=width)
        axs[0].bar(x2, fps, width=width)
        axs[0].grid(axis='y')
        # axs[0].set_xlabel('Crowdedness')
        axs[0].set_ylabel('relative improvement')
        axs[0].spines['left'].set_color('none')
        axs[0].spines['right'].set_color('none')
        axs[0].spines['top'].set_color('none')
        axs[0].spines['bottom'].set_position(('data', 0))

        axs[1].bar(x1, tps1, width=width, label="TP")
        axs[1].bar(x2, fps1, width=width, label="FP")
        axs[1].set_xticks(x)   # x轴坐标长度
        axs[1].set_xticklabels(x_data)
        axs[1].set_xlabel('Crowdedness')
        axs[1].legend()  # 给出图例
        axs[1].grid(axis='y')
        axs[1].spines['left'].set_color('none')
        axs[1].spines['right'].set_color('none')
        axs[1].spines['top'].set_color('none')

        plt.savefig(str(thres) + 'crowdedness'+'.svg', format='svg')
        #plt.show()