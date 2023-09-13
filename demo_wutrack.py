from loguru import logger

import cv2
import numpy as np
import math

import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, visualize
from yolox.utils.visualize2 import plot_tracking
from yolox.utils.visualize2 import plot_tracking_vote
from yolox.utils.visualize2 import plot_wutracking_vote
from yolox.tracker.wu_tracker import WuTracker
from yolox.tracking_utils.timer import Timer

import argparse
import os
import os.path as osp
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".PNG"]  # 加一个大写


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='result')
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        # "--path", default="./videos/20230707/1.mp4", help="path to images or video"
        "--path", default="../datasets/chicken20210828/val", help="path to images or video"
        # "--path", default="/media/z/ae609a98-67c3-41ce-beed-791c5c3bf738/zzy/data/8月28日小鸡视频/麻羽1/VID_20210828_095100.mp4", help="path to images or video"
        # "--path", default="/media/z/ae609a98-67c3-41ce-beed-791c5c3bf738/zzy/data/8月28日小鸡视频/白羽/VID_20210828_093734.mp4", help="path to images or video"
        # "--path", default="/media/z/ae609a98-67c3-41ce-beed-791c5c3bf738/zzy/data/8月28日小鸡视频/白羽2/VID_20210821A1.mp4", help="path to images or video"
        # "--path", default="/media/z/ae609a98-67c3-41ce-beed-791c5c3bf738/zzy/data/8月28日小鸡视频/麻羽2/VID_20210828_161118.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='../exps/example/mot/yolox_m_chicken_half.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='../YOLOX_outputs/yolox_m_chicken_half/best_ckpt.pth.tar', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.7, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.cls_names = ['chick_mayu', 'chick_baiyu']
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = WuTracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        outputs, img_info = predictor.inference(image_name, timer)
        online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size,direction='left2right')
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)

        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)
        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    #write_results(result_filename, results)


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    temp = time.time()
    import datetime
    # 获取当前时间戳（以秒为单位）
    timestamp = int(time.time())
    # 将时间戳转换为日期和时间字符串（默认格式）
    dt_default = datetime.datetime.fromtimestamp(timestamp)

    # 将时间戳转换为自定义格式的日期和时间字符串
    dt_custom = datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d-%H:%M:%S')
    save_folder = os.path.join(
        vis_folder, dt_custom, 'track'
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # 使用WuTracker
    tracker = WuTracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    ##chicken_ID##
    # total_id_top_1 = []
    # total_id_bottom_1 = []
    # total_id_left_1 = []
    # total_id_right_1 = []
    # total_id_top_2 = []
    # total_id_bottom_2 = []
    # total_id_left_2 = []
    # total_id_right_2 = []
    # total_id_top_3 = []
    # total_id_bottom_3 = []
    # total_id_left_3 = []
    # total_id_right_3 = []
    '''
    记录小鸡的方向 表示上下左右
    '''

    direction = ''

    center_points = {}
    total_count = []

    '''byte id'''
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            ''' ignore the frame without detection '''
            if outputs[0] is None:
                continue
            '''get direction'''
            if frame_id == 0 and outputs is not None:
                img_w = img_info['width']
                img_h = img_info['height']
                # 获取第一个检测框的中心坐标
                bbox_first = (int(outputs[0][0, 0]) + int(outputs[0][0, 2])) / 2
                ratio = img_info["ratio"]
                bbox_first /= ratio
                if bbox_first <= img_w / 2:
                    direction = 'left2right'
                else:
                    direction = 'right2left'
            # direction = 'left2right'
            # direction = 'right2left'
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size,
                                            direction)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                # vertical = tlwh[2] / tlwh[3] > 1.6
                # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            timer.toc()
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            '''origin'''
            #online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
            #                          fps=1. / timer.average_time)
            '''count_track_vote'''
            # online_im, total_count, center_points = visualize.plot_wutracking_vote(total_count, center_points, direction,
            #                                                              img_info['raw_img'], online_tlwhs, online_ids,
            #                                                              frame_id=frame_id, fps=1. / timer.average_time)
            '''count_id'''
            # online_im, total_count, center_points = visualize.plot_wutracking_id(total_count, center_points, direction,
            #                                                              img_info['raw_img'], online_tlwhs, online_ids,
            #                                                              frame_id=frame_id, fps=1. / timer.average_time)

            '''count_line'''
            # online_im, total_count, center_points = visualize.plot_wutracking_line(total_count, center_points, direction,
            # img_info['raw_img'], online_tlwhs, online_ids,
            # frame_id=frame_id, fps=1. / timer.average_time)

            '''count_region'''
            online_im, total_count, center_points = visualize.plot_wutracking_region(total_count, center_points, direction,
            img_info['raw_img'], online_tlwhs, online_ids,
            frame_id=frame_id, fps=1. / timer.average_time)

            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    fps = 1. / max(1e-5, timer.average_time)
    # write_count_chicken(save_folder, args.path,
    #                     count_top=[len(total_id_top_1), len(total_id_top_2), len(total_id_top_3)],
    #                     count_bottom=[len(total_id_bottom_1), len(total_id_bottom_2), len(total_id_bottom_3)],
    #                     count_left=[len(total_id_left_1), len(total_id_left_2), len(total_id_left_3)],
    #                     count_right=[len(total_id_right_1), len(total_id_right_2), len(total_id_right_3)],
    #                     d=direction.index(max(direction)))
    write_count_chicken(save_folder, args.path, len(total_count), len(center_points.keys()), fps)

    '''detection'''
    # while True:
    #     if frame_id % 20 == 0:
    #         logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
    #     ret_val, frame = cap.read()
    #     if ret_val:
    #         outputs, img_info = predictor.inference(frame, timer)
    #         result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
    #         if args.save_result:
    #             vid_writer.write(result_frame)
    #         ch = cv2.waitKey(1)
    #         if ch == 27 or ch == ord("q") or ch == ord("Q"):
    #             break
    #     else:
    #         break
    #     frame_id += 1


def write_count_chicken(save_folder, video, count_top, count_bottom, count_left, count_right, d):
    '''
        第一位表示过中间线的鸡的数量，
        第二位表示过三条线的鸡的平均数（A + B + C） / 2
        第三位表示A * 0.25 + B * 0.5 + c * 0.25
        第四位表示取三条线中的最大值
    '''
    chicken_number = [0, 0, 0, 0]
    if d == 0:  # top
        # print('从上到下')
        chicken_number[0] = count_top[1]
        chicken_number[1] = math.ceil(np.mean(count_top))
        chicken_number[2] = math.ceil(count_top[0] * 0.25 + count_top[1] * 0.5 + count_top[2] * 0.25)
        chicken_number[3] = np.max(count_top)

    elif d == 1:  # bottom
        # print('从下到上')
        chicken_number[0] = count_bottom[1]
        chicken_number[1] = math.ceil(np.mean(count_bottom))
        chicken_number[2] = math.ceil(count_bottom[0] * 0.25 + count_bottom[1] * 0.5 + count_bottom[2] * 0.25)
        chicken_number[3] = np.max(count_bottom)

    elif d == 2:  # left
        # print('从左到右')
        chicken_number[0] = count_left[1]
        chicken_number[1] = math.ceil(np.mean(count_left))
        chicken_number[2] = math.ceil(count_left[0] * 0.25 + count_left[1] * 0.5 + count_left[2] * 0.25)
        chicken_number[3] = np.max(count_left)

    elif d == 3:  # right
        # print('从右到左')
        chicken_number[0] = count_right[1]
        chicken_number[1] = math.ceil(np.mean(count_right))
        chicken_number[2] = math.ceil(count_right[0] * 0.25 + count_right[1] * 0.5 + count_right[2] * 0.25)
        chicken_number[3] = np.max(count_right)

    result_fpath = osp.join(save_folder, 'count.csv')

    video_name = video.split('/')
    video_name = video_name[len(video_name) - 1].split('.')
    video_name = video_name[0]

    result_str = '{},{},{},{},{}\n'.format(video_name, chicken_number[0], chicken_number[1],
                                           chicken_number[2], chicken_number[3])

    with open(result_fpath, 'a') as f:
        f.write(result_str)


def write_count_chicken(save_folder, video, count, id_count, fps):
    result_fpath = osp.join(save_folder, 'count.csv')

    video_name = video.split('/')
    video_name = video_name[len(video_name) - 1].split('.')
    video_name = video_name[0]

    result_str = '{},{},{},{}\n'.format(video_name, count, id_count, fps)
    if not os.path.exists(result_fpath):
        with open(result_fpath, 'a') as f:
            f.write('video_name,count,id_count,fps\n')
            f.write(result_str)
    else :
        with open(result_fpath, 'a') as f:
            f.write(result_str)


def main(exp, args):
    torch.cuda.set_device('cuda:0')
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join('..', args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "count_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
