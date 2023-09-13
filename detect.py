import argparse
import sys
from collections import OrderedDict

import numpy as np
import os

import cv2
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from main_win.win import Ui_mainWindow
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, visualize3
from yolox.utils.visualize import plot_tracking
from yolox.utils.visualize import plot_tracking_vote
from yolox.utils.visualize import plot_wutracking_vote
from yolox.tracker.wu_tracker import WuTracker
from yolox.tracking_utils.timer import Timer
import time
from loguru import logger


#  播放类
class PlayDetThread(QThread):
    send_img = pyqtSignal(np.ndarray)  # 检测后的图像
    send_raw = pyqtSignal(np.ndarray)  # 原图像
    send_percent = pyqtSignal(int)
    send_statistic = pyqtSignal(str)  # 数据
    send_msg = pyqtSignal(str)

    def __init__(self):
        super(PlayDetThread, self).__init__()
        self.percent_length = 1000  # progress bar
        self.is_continue = True  # continue/pause
        self.jump_out = False  # jump out of the loop
        self.raw_video_path = ''
        self.out_video_path = ''
        self.percent = 1000  # 进度条
        self.rate = 50

    # 播放过程
    def run(self):
        raw_cap = cv2.VideoCapture(self.raw_video_path)  # 原视频
        out_cap = cv2.VideoCapture(self.out_video_path)  # 检测后结果
        total_frames = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        while True:
            if self.jump_out:
                raw_cap.release()
                out_cap.release()
                self.send_percent.emit(0)
                self.send_msg.emit('Stop')
                break
            raw_ret, raw_img = raw_cap.read()
            out_ret, out_img = out_cap.read()
            if not raw_ret:
                break
            if self.is_continue:
                current_frame += 1
                self.send_percent.emit(current_frame / total_frames * self.percent)
                raw_img = np.array(raw_img)
                out_img = np.array(out_img)
                self.send_raw.emit(raw_img)
                self.send_img.emit(out_img)
                time.sleep(1 / self.rate)
        raw_cap.release()
        out_cap.release()
        self.send_percent.emit(0)
        self.send_msg.emit("Played")


# 检测类
class DetThread(QThread):
    send_statistic = pyqtSignal(dict)  # 数据
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)

    def __init__(self, playThread):
        super(DetThread, self).__init__()
        self.weights = './yolox_l_chicken_half.pth.tar'
        self.current_weight = './yolox_l_chicken_half.pth.tar'
        self.match_thresh = 0.7
        self.track_thresh = 0.3
        self.is_detecting = False
        self.save_fold = './result'
        self.source = None
        self.total = 0
        self.is_finish = False
        self.save_list = []
        self.playThread = playThread
        self.detected_list = []

    def run(self):
        args = make_parser().parse_args()
        total = 0
        if self.source is not None and self.is_detecting is True:
            args.path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            result_statistic = {}
            for i in self.source:
                result_statistic[str(os.path.basename(i))] = '待检测...'
            self.send_statistic.emit(result_statistic)
            for file in self.source:
                if file in self.detected_list:  # 若文件已检测，跳过循环
                    result_statistic[os.path.basename(file)] = '已检测'
                    self.send_statistic.emit(result_statistic)
                    continue
                self.is_detect = True
                args.file = file
                args.match_thresh = self.match_thresh
                args.track_thresh = self.track_thresh
                result_statistic[os.path.basename(file)] = '检测中'
                self.send_statistic.emit(result_statistic)
                sum = self.run_detect(args)
                self.is_detecting = False
                print('{}检测完成...'.format(os.path.basename(file)))
                result_statistic[os.path.basename(file)] = str(sum)
                self.send_statistic.emit(result_statistic)
                total += sum
                self.detected_list.append(file)
                self.save_list.append(os.path.join(args.path, file.split("/")[-1]))
                if len(self.detected_list) == 1:
                    self.send_msg.emit('one_ok')
            txt_result = os.path.join(args.path, 'results.txt')
            result_statistic['总数量'] = str(total)
            with open(txt_result, 'a', encoding='utf-8') as f:
                for key, value in result_statistic.items():
                    f.write(f"{key}:{value}\n")
            f.close()
            self.send_statistic.emit(result_statistic)
            self.is_finish = True
            self.send_msg.emit('Finished')
            print('全部检测完成...')

    @torch.no_grad()
    def run_detect(self, args):
        try:
            args.exp_file = os.path.join('./exps/example/mot/', os.path.basename(self.weights).split('.')[0] + '.py')
            exp = get_exp(args.exp_file)
            torch.cuda.set_device('cuda:0')

            vis_folder = args.path

            if args.conf is not None:
                exp.test_conf = args.conf
            if args.nms is not None:
                exp.nmsthre = args.nms
            if args.tsize is not None:
                exp.test_size = (args.tsize, args.tsize)
            else:
                args.tsize = exp.test_size[-1]

            model = exp.get_model()

            if args.device == "gpu":
                model.cuda()
            model.eval()

            logger.info("Args: {}".format(args))

            if self.current_weight != self.weights:
                args.ckpt = self.weights
                # load the model state dict
                self.current_weight = self.weights

            ckpt_file = self.weights
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

            if args.fuse:
                model = fuse_model(model)

            if args.fp16:
                model = model.half()  # to FP16

            predictor = Predictor(model, exp, trt_file=None, decoder=None, device=args.device, fp16=args.fp16)
            current_time = time.localtime()
            total = imageflow_demo(predictor, vis_folder, current_time, args)
            return total

        except Exception as e:
            self.send_msg.emit('%s' % e)


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
        # if trt_file is not None:
        #     from torch2trt import TRTModule
        #
        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))
        #
        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt
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


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = os.path.join(args.path, args.file.split("/")[-1])
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    logger.info(f"video save_path is {save_path}")

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
        if frame_id % 20 == 0 and frame_id != 0:
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
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']],
                                            (args.tsize, args.tsize), direction)
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
            # online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
            #                          fps=1. / timer.average_time)
            '''count_track_vote'''
            online_im, total_count, center_points = visualize3.plot_wutracking_vote(total_count, center_points,
                                                                                    direction,
                                                                                    img_info['raw_img'], online_tlwhs,
                                                                                    online_ids,
                                                                                    frame_id=frame_id,
                                                                                    fps=1. / timer.average_time)
            '''count_id'''
            # online_im, total_count, center_points = visualize.plot_wutracking_id(total_count, center_points, direction,
            #                                                              img_info['raw_img'], online_tlwhs, online_ids,
            #                                                              frame_id=frame_id, fps=1. / timer.average_time)

            '''count_line'''
            # online_im, total_count, center_points = visualize.plot_wutracking_line(total_count, center_points,
            #                                                                        direction,
            #                                                                        img_info['raw_img'], online_tlwhs,
            #                                                                        online_ids,
            #                                                                        frame_id=frame_id,
            #                                                                        fps=1. / timer.average_time)

            # '''count_region'''
            # online_im, total_count, center_points = visualize.plot_wutracking_region(total_count, center_points, direction,
            # img_info['raw_img'], online_tlwhs, online_ids,
            # frame_id=frame_id, fps=1. / timer.average_time)

            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    fps = 1. / max(1e-5, timer.average_time)
    return len(total_count)


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--file", type=str, default=None, help="file name")
    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default=None, help="path to images or video"
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/example/mot/yolox_l_chicken_half.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.8, type=float, help="test nms threshold")
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
    parser.add_argument("--match_thresh", type=int, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser
