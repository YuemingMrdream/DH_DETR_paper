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
        # gtboxes = self.gtboxes if self.gtboxes is not None else list()

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True)) #按照预测分数排序
        # gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            #每一个dtbox与所有gtbox计算IOU
            # overlap_iou = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] > 0], True)
            overlap_iou = self.box_overlap_opr(dtboxes, dtboxes, True) # ndarray(6,5)[整，score]

        return overlap_iou

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

    def compare(self, thres=0.5, pt_path=None,matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        IOUlist = list()
        SIMlist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                IOU = self.images[ID].compare_caltech(thres)
                # query 的前几
                query = torch.load(pt_path + ID +'.pt',map_location='cpu') #1,1000,256
                q = query[:,:IOU.shape[0],:].repeat(IOU.shape[0],1,1)
                sim = torch.nn.functional.cosine_similarity(q, q.transpose(0,1), dim=2)
                sim = sim.numpy()
                for i in range(0,IOU.shape[0]):
                    for j in range(i+1,IOU.shape[0]):
                        if IOU[i][j]>0:
                            IOUlist.append(str(IOU[i][j]))
                            # IOUlist.append(str(int(IOU[i][j]*10)/10))
                            if sim[i][j]>1:
                                print('warn')
                            SIMlist.append(sim[i][j])
            if len(IOUlist) >20000:
                break

        return IOUlist, SIMlist


if __name__ == "__main__":

    import torch
    import pandas as pd
    import seaborn as sn
    import numpy as np

    gt_path = './annotation_val.odgt'
    dt_path1 = './DH_head/model_dump/two-stage-deformable-6-6/epoch-49.human'
    dt_path2 = './DH_head/eval_dump/two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500/epoch-51.human'
    pt_path1 = r'ori_query/'
    pt_path2 = r'DeHomo_query/'


    database = Database(gt_path, dt_path1, "box", None,mode=0)
    IOU1, SIM1 = database.compare(pt_path = pt_path1)  # gt和dt的匹配

    database = Database(gt_path, dt_path2, "box", None,mode=0)
    IOU2, SIM2 = database.compare(pt_path = pt_path2)  # gt和dt的匹配

    # SIM1 = np.vstack((IOU1[0:20000], SIM1[0:20000],['Homo query']*20000)).transpose(1, 0)
    # SIM2 = np.vstack((IOU2[0:20000], SIM2[0:20000],['De-Homo query']*20000)).transpose(1,0)
    SIM1 = np.vstack((IOU1[0:5000], SIM1[0:5000],['Homo query']*5000)).transpose(1, 0)
    # SIM2 = np.vstack((IOU2[0:5000], SIM2[0:5000],['De-Homo query']*5000)).transpose(1,0)

    # x = np.vstack((SIM1,SIM2))
    x = SIM1

    data = pd.DataFrame(x)
    data.to_csv('compare_SIM.csv')




    sn.set_style("whitegrid")

    tips = pd.read_csv('compare_SIM.csv')
    tips.columns = ['_', 'Iou', 'Cosine Similarity',' ']

    # # 箱型图+++++++++++++++++++
    # sn.set_theme(style="ticks", palette="pastel")
    # # sn.boxplot(x='Iou', y='SIM', palette=["m", "g"] ,data=tips)
    # fig = sn.boxplot(x='Iou',y='Cosine Similarity', hue=' ', palette=["g","m"] ,data=tips,fliersize=0)
    # # fig = sn.boxplot(x='Iou', y='SIM', palette=["g"] ,data=tips)
    # # plt.legend( loc='lower center')
    # sn.despine(offset=10, trim=True)
    # # 箱型图+++++++++++++++++++


    # # 大提琴图、
    # plt.figure(dpi=1000, figsize=(10, 5))
    # sn.set_theme(style="whitegrid")
    # fig = sn.violinplot(data=tips,x='Iou', y='Cosine Similarity', hue=' ',scale='count',cut=0, palette=["m", "g"],linewidth=0
    #                     # ,split=True
    #                # palette={"Yes": "b", "No": ".85"}
    #               )
    # sn.despine(left=True)




    # 分组的线性回归图+++++++++
    fig = sn.jointplot(x='Iou', y='Cosine Similarity', data=tips, markers=["x", "o"], palette='Set1')
    plt.xlim((0, 1))  # 限制x的值为[0,20]
    plt.ylim((0.6, 1))  # 限制y的值为[0,1]
    plt.savefig("visual/SIMiou5.svg", dpi=1000)
    # 分组的线性回归图+++++++++



    # _fig = fig.get_figure()
    # _fig.savefig(r"visual/SIMiou5.png", dpi=1000, bbox_inches='tight')

    plt.show()
