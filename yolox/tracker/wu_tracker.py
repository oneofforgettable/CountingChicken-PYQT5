import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
            self.track_id = self.next_id()
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched id
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        if not self.is_activated:
            self.track_id = self.next_id()
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class WuTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size, direction):
        self.frame_id += 1
        # print("************** Number  ", self.frame_id ," Frame**************")
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        '''
        根据检检测框出现的位置将将检测框分为3类，分别是：
            出生区：检测框出现在小鸡前进方向的前1/3段视频视野内
            生长区：检测框出现在小鸡前进方向的1/3到2/3段视频视野内
            死亡区：检测框出现在小鸡前进方向的后1/3视频视野内
        '''
        newDetection = []
        growDetection = []
        dieDetection = []


        img_h, img_w = img_info[0], img_info[1]


        areaSize_first = img_w / 3
        areaSize_second = areaSize_first * 2

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            center = (output_results[:,0] + output_results[:,2])/2
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            center = (output_results[:, 0] + output_results[:, 2]) / 2

        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        center /= scale

        ##只保留置信度大于0.1的检测框
        remain_inds = scores >= 0.1
        dets = bboxes[remain_inds]
        center = center[remain_inds]
        scores_keep = scores[remain_inds]
        # print('检测框中心x坐标：',center)
        # print('检测框置信度', scores_keep)
        #ToDO
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []


        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 已有的全部轨迹（不包含已经remove的轨迹）
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 将已有的轨迹和全部的检测框进行匹配
        STrack.multi_predict(unconfirmed)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # print("*******第一次IOU匹配，将所有的轨迹和检测框进行匹配*******")
        # print("matches:", matches)
        # print("matches:", matches[:,1])
        # print("u_track", u_track)
        # print("u_detection", u_detection)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        '''
            根据检检测框出现的位置将将检测框分为3类，分别是：
                出生区：检测框出现在小鸡前进方向的前1/3段视频视野内
                生长区：检测框出现在小鸡前进方向的1/3到2/3段视频视野内
                死亡区：检测框出现在小鸡前进方向的后1/3视频视野内
        '''
        if direction == 'right2left':
            for idx in u_detection:
                if center[idx] >= areaSize_second:
                    newDetection.append(idx)
                elif center[idx] <= areaSize_first:
                    dieDetection.append(idx)
                else:
                    growDetection.append(idx)
        elif direction == 'left2right':
            for idx in u_detection:
                if center[idx] <= areaSize_first:
                    newDetection.append(idx)
                elif center[idx] >= areaSize_second:
                    dieDetection.append(idx)
                else:
                    growDetection.append(idx)
        else:
            newDetection = []
            growDetection = []
            dieDetection = []
        # print("******第一次IOU匹配后的剩余检测框分为3类******")
        # print('newDetection:', newDetection)
        # print('growDetection:', growDetection)
        # print('dieDetection:', dieDetection)
        ''' 第二次IOU匹配，将所有匹配失败的检测框（位于生长区或死亡区）和所有匹配失败的轨迹，再进行一次匹配（thresh比第一次匹配大0.25） '''
        r_tracked_stracks = [strack_pool[i] for i in u_track]
        detections_in_growOrDie = [detections[i] for i in (growDetection or dieDetection)]
        # print('detections_in_growOrDie:',detections_in_growOrDie)
        dists = matching.iou_distance(r_tracked_stracks, detections_in_growOrDie)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.args.match_thresh+0.25)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_in_growOrDie[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        '''只要新的检测框在两帧之内能够匹配成功，将其激活并分配ID'''
        detections = [detections[i] for i in newDetection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            if self.frame_id - track.end_frame > 3:
                track.mark_removed()
                removed_stracks.append(track)
            else :
                activated_starcks.append(track)


        """ 将生长区中没有匹配的检测框都创建为未激活的tracker"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
