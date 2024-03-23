import sys
from common import *
def computeJaccard(fpath, save_path ='results.md'):

    assert os.path.exists(fpath)
    records = load_func(fpath)

    GT = load_func(config.anno_file)
    fid = open(save_path, 'a')
    for i in range(1, 10):
        score_thr = 1e-1 * i
        results = common_process(worker, records, 4, GT, score_thr, 0.5)
        line = strline(results)
        line = 'score_thr:{:.3f}, '.format(score_thr) + line
        print(line)
        fid.write(line + '\n')
        fid.flush()
    fid.close()

def filter_boxes(result_queue, records, score_thr, nms_thr):

    assert np.logical_and(score_thr >= 0., nms_thr > 0.)
    for i, record in enumerate(records):

        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] > score_thr
        if flag.sum() < 1:
            result_queue.put_nowait(None)
            continue

        cls_dets = dtboxes[flag]
        keep = nms(np.float32(cls_dets), nms_thr)
        res = record
        res['dtboxes'] = boxes_dump(cls_dets[keep])
        result_queue.put_nowait(res)

def compute_iou_worker(result_queue, records, score_thr):

    for record in records:
        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        result = record
        result['dtboxes'] = list(filter(lambda rb: rb['score'] >= score_thr, record['dtboxes']))
        result_queue.put_nowait(result)

def computeIoUs(fpath, score_thr = 0.1, save_file='record.txt', exp='human'):
    
    print('Processing {}'.format(osp.basename(fpath)))
    name = os.path.basename(fpath)

    records = load_func(fpath)
    results = common_process(compute_iou_worker, records, 4, score_thr)
    
    fpath = exp + '.human'
    save_results(results, fpath)
    mAP, mMR = compute_mAP(fpath)

    fid = open(save_file, 'a')
    fid.write('{}\ndtboxes:\n'.format(name))
    print('{}\ndtboxes:\n'.format(name))
    line = 'mAP:{:.4f}, mMR:{:.4f}, '.format(mAP, mMR)
    print(line)
    fid.write(line + '\n')
    fid.close()
    computeJaccard(fpath, save_path=save_file)
    os.remove(fpath)

def test_unit():

    fpath = osp.join(config.eval_dir, 'epoch-30.human')
    records = load_func(fpath)
    save_path = 'nms.md'
    
    score_thr = 0.1
    for i in range(1, 9):
        nms_thr = 0.1 * i
        results = common_process(filter_boxes, records, 16, score_thr, nms_thr)
        fpath = 'mountain.human'
        save_results(results, fpath)
        mAP, mMR = compute_mAP(fpath)
        line = 'score_thr:{:.1f}, mAP:{:.4f}, mMR:{:.4f}'.format(score_thr, mAP, mMR)
        print(line)
        fid = open(save_path, 'a')
        fid.write(line + '\n')
        fid.close()
        computeJaccard(fpath, save_path)

def eval_all():
    score_thr = 0.05
    save_file = f'record_{sys.argv[1]}.txt'
    fpath = osp.join(config.eval_dir, sys.argv[1], 'epoch-{}.human'.format(sys.argv[2]))
    computeIoUs(fpath, score_thr, save_file, sys.argv[1])

def random_variables(gtboxes, height, width):

    alpha = 0.03
    targets = gtboxes
    boxes = xyxy_to_cxcywh(gtboxes)
    scale = np.array([width, height, width, height]).reshape(-1, 4)
    boxes /= scale
    n = boxes.shape[0]
    
    offsets = np.random.randn(12*n) * alpha
    offsets = offsets.reshape(-1, 3, 4)
    proposals = np.expand_dims(boxes, axis=1) + offsets
    
    proposals = proposals * scale.reshape(-1, 1, 4)
    proposals = proposals.reshape(-1, proposals.shape[2])
    proposals = cxcywh_to_xyxy(proposals)
    
    return proposals, targets

def limit_in_boxes(proposals, gtboxes, ratio = 0.3):

    assert proposals.shape[1] > 3 and gtboxes.shape[1] > 3
    local_gtboxes = compute_pos_area(gtboxes, ratio)
    n, k = proposals.shape[0], local_gtboxes.shape[0]
    pcs = xyxy_to_cxcywh(proposals)
    pcs = np.expand_dims(pcs[..., :2], axis=1).repeat(k, axis = 1)
    t = np.expand_dims(local_gtboxes, axis = 0).repeat(n, axis=0)
    
    lt = pcs[..., 0] - t[..., 0]
    tp = pcs[..., 1] - t[..., 1]
    rt = t[..., 2] - pcs[..., 0]
    bm = t[..., 3] - pcs[..., 1]
    
    dist = np.stack([lt, tp, rt, bm], axis=2).min(axis = 2) >= 0.1
    
    return dist

def _compute_excluding_overlap(proposals, boxes):
    
    
    overlaps = [compute_iou_matrix(proposals, bs) for bs in boxes]
    matches = [linear_sum_assignment(-overlap) for overlap in overlaps]
    ious = [overlap[m] for overlap, m in zip(overlaps, matches)]
    ious = np.array([iou.sum(-1) for iou in ious])
    return ious

def compute_optimal_matching(dtboxes, gtboxes):

    assert dtboxes.shape[1] > 3 and gtboxes.shape[1] > 3
    groups = nms_groups(gtboxes, 0.5)
    boxes = [gtboxes[g] for g in groups]
    
    proposals = dtboxes.reshape(-1, 3, dtboxes.shape[1])
    n, k, c = proposals.shape
    
    results = [_compute_excluding_overlap(proposals[i], boxes) for i in range(n)]
    costs = np.stack(results, axis = 0)
    matches = linear_sum_assignment(-costs)
    
    return matches, boxes


# # huang
def draw_each_box(draw_img ,bbox, label_color, score, points):
    color = tuple(map(int,np.random.randint(0, high=256, size=(3,))))

    label = str(score)
    box_color = color
    label_color = color

    # 绘制框   左上坐标和右下坐标
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=box_color, thickness=1)

    # 绘制点：格式为：coordinates=[[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]
    point_size = 1
    point_color =color  # BGR
    thickness = 20
    for coor in points[0]:
        cv2.circle(draw_img, (int(coor[0]), int(coor[1])), point_size, box_color, thickness)
        # 绘制椭圆
        cv2.ellipse(draw_img, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), 0, 0, 360, box_color, thickness=2)

    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1
                      )
        # cv2.putText(draw_img, label,
        #             (bbox[0], bbox[1] + labelSize + 3),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (0, 0, 0),
        #             thickness=1
        #             )
    else:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
# # huang
def draw_box_corner(draw_img,bbox,length,corner_color):
    # Top Left
    cv2.line(draw_img, (bbox[0], bbox[1]), (bbox[0] + length, bbox[1]), corner_color, thickness=2)
    cv2.line(draw_img, (bbox[0], bbox[1]), (bbox[0], bbox[1] + length), corner_color, thickness=2)
    # Top Right
    cv2.line(draw_img, (bbox[2], bbox[1]), (bbox[2] - length, bbox[1]), corner_color, thickness=2)
    cv2.line(draw_img, (bbox[2], bbox[1]), (bbox[2], bbox[1] + length), corner_color, thickness=2)
    # Bottom Left
    cv2.line(draw_img, (bbox[0], bbox[3]), (bbox[0] + length, bbox[3]), corner_color, thickness=2)
    cv2.line(draw_img, (bbox[0], bbox[3]), (bbox[0], bbox[3] - length), corner_color, thickness=2)
    # Bottom Right
    cv2.line(draw_img, (bbox[2], bbox[3]), (bbox[2] - length, bbox[3]), corner_color, thickness=2)
    cv2.line(draw_img, (bbox[2], bbox[3]), (bbox[2], bbox[3] - length), corner_color, thickness=2)
# # huang
def draw_each_box_rel(draw_img ,bbox, label_color, score, rel=None, points=None):
    color = tuple(map(int,np.random.randint(0, high=256, size=(3,))))

    # label = str(score)+':'+str(np.round(rel, decimals=3))+':'
    label = str(score)
    box_color = color
    label_color = color

    # 绘制框   左上坐标和右下坐标
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=box_color, thickness=1)
    draw_box_corner(draw_img, bbox, 30, box_color)
    # # 绘制点：格式为：coordinates=[[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]
    # point_size = 1
    # point_color =color  # BGR
    # thickness = 10
    # for coor in points:
    #     cv2.circle(draw_img, (int(coor[0]), int(coor[1])), point_size, box_color, thickness)
    #     # 绘制椭圆
    #     cv2.ellipse(draw_img, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), 0, 0, 360, box_color, thickness=2)

    # 绘制置信度
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1
                      )
        # cv2.putText(draw_img, label,
        #             (bbox[0], bbox[1] + labelSize + 3),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (0, 0, 0),
        #             thickness=1
        #             )
    else:
        # cv2.rectangle(draw_img,
        #               (bbox[0], bbox[1] - labelSize[1] - 3),
        #               (bbox[0] + labelSize[0], bbox[1] - 3),
        #               color=label_color,
        #               thickness=-1
        #               )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color=label_color,
                    thickness=1
                    )
# 黄
def draw_xt(xt: np.ndarray, image: np.ndarray, color: Color,line_width: int, rel = None,points = None):
    assert image is not None
    score = np.round(xt[:, 4], decimals=3)
    xt = np.int32(np.round(xt[:, :4]))   # 第5个维度是置信度
    nr_xt = xt.shape[0]
    for i in range(nr_xt):
        # box = Box(xt[i, :])
        if rel ==None:
            draw_each_box_rel(image, xt[i, :], color, score[i])
        else:
            draw_each_box_rel(image, xt[i, :], color, score[i], rel = rel[0][i], points=points[0][i])

# huang
def recover_(record):
    assert 'dtboxes' in record
    if len(record['dtboxes']) < 1:
        return np.empty([0, 5])
    dtboxes = np.vstack([np.hstack((rb['box'], rb['score'])) for rb in record['dtboxes']])
    assert dtboxes.shape[1] >= 4
    dtboxes[:, :2] -= 0.5*dtboxes[:, 2:4]
    dtboxes[:, 2:4] += dtboxes[:, :2]
    return dtboxes

def paint_images_heatmap(record,points=None, rel = None):
    visDir = 'Vismap01'
    ensure_dir(visDir)
    with open(r'visual/cur_filename.txt', "r", encoding='utf-8') as f:
        img_path = f.read()  # 读取文本
    imgpath = osp.join(config.imgDir2, img_path + '.jpg')
    # imgpath = osp.join(r'E:\00Experiment\_Datasets\CrowdHuman\CrowdHuman_val\Images', record['ID'] + '.jpg')
    image = cv2.imread(imgpath)
    height, width = image.shape[:2]
    dtboxes = recover_(record)  # (左上x,y,右下 x,y
    dtboxes =  dtboxes*[width,height,width,height,1]
    flag = dtboxes[:, 4] >= 0.2 # 置信度 >0.2的你
    dtboxes = dtboxes[flag]  # 取出检测出来 大于置信度0.2 的
    # leverage the center priors to ensure the valid bounding boxes located in the center area of the targets
    draw_xt(dtboxes, image, Color.Green, 2,rel = rel,points= points)
    fpath = osp.join(visDir,  img_path + '.png')
    cv2.imwrite(fpath, image)



# huang
def paint_test_images(record, points=None, rel = None):
    visDir = 'Vismap01'
    ensure_dir(visDir)
    imgpath = osp.join(config.imgDir, record['ID'] + '.jpg')
    # imgpath = osp.join(r'E:\00Experiment\_Datasets\CrowdHuman\CrowdHuman_val\Images', record['ID'] + '.jpg')
    image = cv2.imread(imgpath)
    height, width = image.shape[:2]
    dtboxes = recover_dtboxes(record)
    flag = dtboxes[:, 4] >= 0.2 # 置信度 >0.2的你
    dtboxes = dtboxes[flag]  # 取出检测出来 大于置信度0.2 的
    # leverage the center priors to ensure the valid bounding boxes located in the center area of the targets
    draw_xt(dtboxes, image, Color.Green, 2,rel = rel,points= points)
    fpath = osp.join(visDir, record['ID'] + '.png')
    cv2.imwrite(fpath, image)



def paint_images():

    fpath = r'DH_head/eval_dump/two-stage-deformable-6-6/epoch-49.human'
    # fpath = r'DH_head/eval_dump/two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500/epoch-51.human'
    # fpath = r'/root/autodl-tmp/Iter-deformable/CITY_output/eval_dump/CITYPERSON_1_0/epoch-34.human'
    records = load_func(fpath)
    
    visDir = 'detrick/two-stage-deformable-6-6'
    # visDir = 'detrick/two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500'
    ensure_dir(visDir)
    count = 0
    from tqdm import tqdm
    for i, record in tqdm(enumerate(records)) :
        imgpath = osp.join(r"E:\00Experiment\_Datasets\CrowdHuman\CrowdHuman_val\Images", record['ID'] + '.jpg')
        # imgpath = osp.join(config.imgDir, record['ID'] + '.png')

        image = cv2.imread(imgpath)
        # height, width = image.shape[:2]
        
        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= 0.3 # 置信度 >0.2的你

        dtboxes = dtboxes[flag]     # 取出检测出来 大于置信度0.2 的
        # leverage the center priors to ensure the valid bounding boxes located in the center area of the targets
        draw_xt(dtboxes, image, Color.Green,  2)
        fpath = osp.join(visDir, record['ID'] + '.png')
        cv2.imwrite(fpath, image)
        # count += 1
        # if count> 1000:
        #     break


def capture_target_wkr(result_queue, records, tgts, GT, score_thr):

    assert isinstance(tgts, dict)
    for record in records:

        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        dtboxes = recover_dtboxes(record)
        flag = dtboxes[:, 4] >= score_thr
        result = record

        if flag.sum() < 1:
            result_queue.put_nowait(None)
            continue

        gtboxes, ignores = recover_gtboxes(GT[tgts[record['ID']]])
        gtboxes = gtboxes[~ignores]
        matches = compute_JC(dtboxes[flag], gtboxes, 0.5)

        # rows = np.array([i for i, _ in matches])
        cols = np.array([i for _, i in matches])
        
        rest = dtboxes[~flag]
        indices = np.array(list(set(np.arange(gtboxes.shape[0]))  - set(cols)))
        res, auxi = None, []

        keep = dtboxes[flag]
        if indices.size:

            matches = compute_JC(rest, gtboxes[indices], 0.5)
            
            if len(matches):
                rs = np.array([i for i, _ in matches])
                keep = np.vstack([dtboxes[flag], rest[rs]])
                auxi = boxes_dump(rest[rs])
                

        result['dtboxes'] = boxes_dump(dtboxes[flag])
        result['auxi'] =  auxi
        result['main'] = boxes_dump(dtboxes[flag])
        result_queue.put_nowait(result)

def compute_auxi():

    fpath = 'mountain.human'
    records = load_func(fpath)
    visDir = 'vis_images'
    ensure_dir(visDir)
    
    total, n = 0, 0
    res = []
    for i, record in enumerate(records):
        auxi = record['auxi']
        if len(auxi) < 1:
            continue
        dtboxes = recover_dtboxes(record)

        flag = dtboxes[:, 4] >= 0.5
        boxes = dtboxes[flag]
        keep = nms(np.float32(boxes), 0.5)
        if len(keep) >= boxes.shape[0]:
            continue
        
        indices = np.array(list(set(np.arange(boxes.shape[0])) - set(keep)))
        main, auxi = dtboxes[keep], dtboxes[indices]
        overlaps = compute_iou_matrix(auxi, main)
        index = np.argmax(overlaps, axis = 1)

        bravo = main[index]
        n += 1
        total += bravo.shape[0]
        n += 1

    
    print('total:{}, n:{}'.format(total, n))

if __name__ == '__main__':

    # eval_all()
    paint_images()
    #fpath = 'output/eval_dump/epoch-0.human'
    #ssert osp.exists(fpath)
    #records = load_func(fpath)

    # pdb.set_trace()
    #GT = load_func(config.anno_file)
    #tgts = {}
    #_ = [tgts.update({rb['ID']:i}) for i, rb in enumerate(GT)]
    #tgts.update({})

    #results = common_process(capture_target_wkr, records, 16, tgts, GT, 0.5)

    # dtboxes = [recover_boxes(r, 'main') for r in results]
    # auxies = [recover_boxes(r, 'auxi') for r in results]
    #fpath = 'mountain.human'
    #save_results(results, fpath)
    
    #computeIoUs(fpath,0., save_file='record.md')
    
