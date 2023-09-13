from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QDialog, QListWidget, QVBoxLayout, \
    QTableWidgetItem
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from CustomMessageBox import MessageBox
import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
from detect import DetThread, PlayDetThread


class ListDialog(QDialog):
    def __init__(self, source):
        super(ListDialog, self).__init__()

        self.setWindowTitle("选择文件")
        self.source = source
        self.ListWidget = QListWidget()
        self.ListWidget.addItems(self.source)
        layout = QVBoxLayout()
        layout.addWidget(self.ListWidget)
        self.setLayout(layout)
        self.setFixedSize(500, 300)
        self.ListWidget.itemClicked.connect(self.item_clicked)
        self.select_video = None

    def item_clicked(self):
        selected_items = self.ListWidget.selectedItems()
        selected_values = [item.text() for item in selected_items]
        self.select_video = selected_values
        self.accept()

    def get_select_video(self):
        return self.select_video


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        #  1.设置可放缩窗口
        self.setWindowFlags(Qt.CustomizeWindowHint)

        #   2.窗口不可放缩
        # self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
        #                     | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # 显示窗口
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.tar')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)

        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        self.model_type = self.comboBox.currentText()

        #  选择文件
        self.fileButton.clicked.connect(self.open_file)

        #  播放视频线程
        self.play_thread = PlayDetThread()
        self.play_thread.percent_length = self.progressBar.maximum()
        self.play_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))  # 原视频
        self.play_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))  # 输出视频
        self.play_thread.send_msg.connect(lambda x: self.show_msg(x))  # 底下栏传送信息
        self.play_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))  # 进度条
        self.runButton.clicked.connect(self.play_or_continue)
        self.stopButton.clicked.connect(self.stop)

        #  检测
        self.det_thread = DetThread(self.play_thread)
        self.det_thread.weights = "./models/%s" % self.model_type
        self.det_thread.source = None
        # self.det_thread.percent_length = self.progressBar.maximum()
        # self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))  # 原视频
        # self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))  # 输出视频
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        # self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        # self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        #  确定检测
        self.detectButton.clicked.connect(self.detect)

        #  选择模型
        self.comboBox.currentTextChanged.connect(self.change_model)
        #  改变检测器参数
        self.MTSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'MTSpinBox'))
        self.MTSlider.valueChanged.connect(lambda x: self.change_val(x, 'MTSlider'))
        self.T2SpinBox.valueChanged.connect(lambda x: self.change_val(x, 'T2SpinBox'))
        self.T2Slider.valueChanged.connect(lambda x: self.change_val(x, 'T2Slider'))
        #  是否保存
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()
        # 列表
        self.selectButton.clicked.connect(self.selectVideo)

    # 列表选择视频播放
    def selectVideo(self):
        #if self.det_thread.is_finish:
        if self.det_thread.detected_list is not None:
            self.play_thread.is_continue = False
            video_list = [str(file).split('/')[-1] for file in self.det_thread.detected_list]  # 只展示文件名
            list_dialog = ListDialog(video_list)
            list_dialog.exec_()
            file = list_dialog.get_select_video()  # 必须声明在exec_()后面，不然获取不到值
            if file:
                if isinstance(file, list):
                    file = file[0]
                index = video_list.index(file)
                raw_video_path = self.det_thread.detected_list[index]
                out_video_path = self.det_thread.save_list[index]
                self.play_thread.raw_video_path = raw_video_path
                self.play_thread.out_video_path = out_video_path
                self.runButton.setChecked(True)
                self.play_or_continue(True)

    # 检测按钮触发事件
    def detect(self):
        if self.det_thread.source is None:
            self.show_msg("请选择文件")
        else:
            if self.detectButton.isChecked():
                self.det_thread.is_detecting = True
                print("检测...")
                if not self.det_thread.isRunning():
                    self.det_thread.start()
                # if self.det_thread.is_finish:
                # self.runButton.setChecked(True)  # 设置runButton为选定状态
            self.det_thread.jump_out = False

    #  搜索权重文件
    def search_pt(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.tar')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    #  检测后结果保存
    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    #  加载参数
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            match_thresh = 0.7
            track_thresh = 0.3
            check = 0
            savecheck = 0
            new_config = {"match_thresh": match_thresh,
                          "track_thresh": track_thresh,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                match_thresh = 0.7
                track_thresh = 0.3
                check = 0
                savecheck = 0
            else:
                match_thresh = config['match_thresh']
                track_thresh = config['track_thresh']
                check = config['check']
                savecheck = config['savecheck']
        self.MTSpinBox.setValue(match_thresh)
        self.T2SpinBox.setValue(track_thresh)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    #  更改match_thresh、track_thresh、rate的值
    def change_val(self, x, flag):
        if flag == 'MTSpinBox':
            self.MTSlider.setValue(int(x * 100))
        elif flag == 'MTSlider':
            self.MTSpinBox.setValue(x / 100)
            self.det_thread.match_thresh = x / 100
        elif flag == 'T2SpinBox':
            self.T2Slider.setValue(int(x * 100))
        elif flag == 'T2Slider':
            self.T2SpinBox.setValue(x / 100)
            self.det_thread.track_thresh = x / 100
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        self.qtimer.start(6000)

    def show_msg(self, msg):
        # self.runButton.setChecked(Qt.Unchecked)  # Qt.Unchecked 是 QtCore.Qt 模块中的一个属性，用于表示复选框或按钮处于未选中状态
        self.statistic_msg(msg)
        if msg == "Finished":
            print("检测完成，保存文件...")
            self.play_thread.raw_video_path = self.det_thread.source[0]
            self.play_thread.out_video_path = self.det_thread.save_list[0]
            self.runButton.setChecked(True)
            self.play_or_continue(True)
            self.saveCheckBox.setEnabled(True)
        elif msg == 'Played':
            self.runButton.setChecked(Qt.Unchecked)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./models/%s" % self.model_type
        print("更改模型为：{}".format(x))
        self.statistic_msg('Change model to %s' % x)

    #  输入文件
    def open_file(self):
        try:
            print("选择检测文件")
            config_file = 'config/fold.json'
            # config = json.load(open(config_file, 'r', encoding='utf-8'))
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()  # 获取当前程序的工作目录
            names, _ = QFileDialog.getOpenFileNames(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                                    "*.jpg *.png)")
            temp = []
            if names:
                print("共{}个文件：{}".format(len(names), names))
                self.det_thread.source = names
                for name in names:
                    temp.append(os.path.basename(name))
                self.statistic_msg('载入：{} 共{}个文件'.format(temp, len(names)))
                config['open_fold'] = os.path.dirname(names[-1])
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
            else:
                self.statistic_msg("请选择文件")
        except Exception as e:
            self.statistic_msg(e)
            print("文件选择异常:{}".format(e))

    #  窗口最大化或者恢复正常
    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 播放或者继续
    def play_or_continue(self, flag):
        if self.det_thread.source is None:
            self.show_msg("没有文件")
        else:
            self.play_thread.jump_out = False
            if self.runButton.isChecked():
                self.play_thread.is_continue = True
                if self.det_thread.is_finish:
                    print("开始播放...")
                    if flag and not self.play_thread.isRunning():
                        self.play_thread.start()
            else:
                self.play_thread.is_continue = False
                self.statistic_msg('Pause')

    def stop(self):
        self.play_thread.jump_out = True
        self.runButton.setChecked(Qt.Unchecked)
        self.raw_video.clear()
        self.out_video.clear()

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    # 图像流
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            self.resultWidget.setColumnCount(2)
            self.resultWidget.setShowGrid(True)
            self.resultWidget.verticalHeader().setVisible(False)
            # self.resultWidget.horizontalHeader().setVisible(False)
            self.resultWidget.setHorizontalHeaderLabels(['文件', '结果'])
            if statistic_dic is not None:
                self.resultWidget.setRowCount(len(statistic_dic))
                num = 0
                for i in statistic_dic:
                    file = QTableWidgetItem(i)
                    result = QTableWidgetItem(statistic_dic[i])
                    self.resultWidget.setItem(num, 0, file)
                    self.resultWidget.setItem(num, 1, result)
                    self.resultWidget.resizeColumnsToContents()
                    num += 1

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.play_thread.jump_out = True
        # config_file = 'config/setting.json'
        # config = dict()
        # config['match_thresh'] = self.MTSpinBox.value()
        # config['track_thresh'] = self.T2SpinBox.value()
        # config['rate'] = self.rateSpinBox.value()
        # config['check'] = self.checkBox.checkState()
        # config['savecheck'] = self.saveCheckBox.checkState()
        # config_json = json.dumps(config, ensure_ascii=False, indent=2)
        # with open(config_file, 'w', encoding='utf-8') as f:
        #     f.write(config_json)
        # sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
