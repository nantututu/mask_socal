# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\project.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import argparse
from email.mime import image
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from src.sort import *
from src.deepsocial import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QCoreApplication

def VisualiseResult(_Map, e):
    Map = np.uint8(_Map)
    histMap = e.convrt2Image(Map)
    visualBird = cv2.applyColorMap(np.uint8(_Map), cv2.COLORMAP_JET)
    visualMap = e.convrt2Image(visualBird)
    visualShow = cv2.addWeighted(e.original, 0.7, visualMap, 1 - 0.7, 0)

    return visualShow, visualBird, histMap


def ColorGenerator( seed=1, size=10):
    np.random.seed = seed
    color = dict()
    for i in range(size):
        h = int(np.random.uniform() * 255)
        color[i] = h
    return color


# list11=[]
# for i in range(100):
#     list11.append(i)
def extract_humans(detections):
    detetcted = []
    if len(detections) > 0:  # At least 1 detection in the image and check detection presence in a frame
        idList = []
        id = 0
        for label, confidence, bbox in detections:
            # print(bbox)
            if label == 0:
                xmin, ymin, xmax, ymax = bbox2points(bbox)
                id += 1
                if id not in idList: idList.append(id)
                detetcted.append([int(xmin), int(ymin), int(xmax), int(ymax), idList[-1]])
    return np.array(detetcted)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def centroid(detections, image, calibration, _centroid_dict, CorrectionShift, HumanHeightLimit):
    e = birds_eye(image.copy(), calibration)
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:
        for d in detections:
            p = int(d[4])
            now_present.append(p)
            xmin, ymin, xmax, ymax = d[0], d[1], d[2], d[3]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w / 2
            y = ymax - h / 2
            if h < HumanHeightLimit:
                overley = e.image
                bird_x, bird_y = e.projection_on_bird((x, ymax))
                if CorrectionShift:
                    if checkupArea(overley, 1, 0.25, (x, ymin)):
                        continue
                e.setImage(overley)
                center_bird_x, center_bird_y = e.projection_on_bird((x, ymin))
                centroid_dict[p] = (
                    int(bird_x), int(bird_y),
                    int(x), int(ymax),
                    int(xmin), int(ymin), int(xmax), int(ymax),
                    int(center_bird_x), int(center_bird_y))

                _centroid_dict[p] = centroid_dict[p]
    return _centroid_dict, centroid_dict, e.image

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        parser = argparse.ArgumentParser()
        parser.add_argument('--Mask', action='store_true',  default=1, help='mask')
        parser.add_argument('--weights_mask', nargs='+', type=str, default=ROOT / 'weights/mask_yolov5n-sim.onnx', help='model path(s)')
        parser.add_argument('--data_mask', type=str, default=ROOT / 'data/mask.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5n-sim.onnx', help='model path(s)')
        parser.add_argument('--source', type=str, default="./Images/OxfordTownCentreDataset.avi", help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true',default=False,help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true',default=True, help='do not save images/videos')
        parser.add_argument('--classes', nargs='+',type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exc', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        self._trackMap = np.zeros((360,640,3), dtype=np.uint8)#(1080,1920,3)
        self._crowdMap = np.zeros((360,640), dtype=np.int) 
        self.colorPool = ColorGenerator(size = 3000)
        self._centroid_dict = dict()
        self._numberOFpeople = list()
        self._greenZone = list()
        self._redZone = list()
        self._yellowZone = list()
        self._final_redZone = list()
        self._relation = dict()
        self._couples = dict()
        self._allPeople = 0
        self._counter = 1
        self.frame = 0
        self.CouplesDetection    = 1  
        self.DTC                 = 0              # Detection, Tracking and Couples
        self.SocialDistance      = 1
        self.CrowdMap            = 0
        self.ViolationDistForIndivisuals = 10
        self.ViolationDistForCouples     = 12
        ####
        self.CircleradiusForIndivsual    = 5
        self.CircleradiusForCouples      = 7
        ######################## 
        self.MembershipDistForCouples    = (7 , 7) # (Forward, Behind) per Pixel
        self.MembershipTimeForCouples    = 30       # Time for considering as a couple (per Frame)----------------------------------
        ######################## (0:OFF/ 1:ON)
        self.CorrectionShift  = 1                  # Ignore people in the margins of the video
        self.HumanHeightLimit = 90
        # ViolationDistForIndivisuals = 28
        # ViolationDistForCouples     = 31
        # ####
        # CircleradiusForIndivsual    = 14
        # CircleradiusForCouples      = 17
        # ########################
        # MembershipDistForCouples    = (16 , 10) # (Forward, Behind) per Pixel
        # MembershipTimeForCouples    = 30       # Time for considering as a couple (per Frame)----------------------------------
        # ######################## (0:OFF/ 1:ON)
        # CorrectionShift  = 1                  # Ignore people in the margins of the video
        # HumanHeightLimit = 200                  # Ignore people with unusual heights
        ########################
        self.Transparency= 0.7
        # calibration      = [[180,162],[618,0],[552,540],[682,464]]
        self.calibration= [[90,81],[309,0],[276,270],[341,232]]
        self.source = str(self.opt.source)
        self.save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.opt.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(self.opt.weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data, fp16=self.opt.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size
        if self.opt.Mask:
            self.device_mask = select_device(self.device)
            self.model_mask = DetectMultiBackend(self.opt.weights_mask, device=self.device_mask, dnn=self.opt.dnn, data=self.opt.data_mask, fp16=self.opt.half)
            self.stride_mask, self.names_mask, self.pt_mask = self.model_mask.stride, self.model_mask.names, self.model_mask.pt
        self.mot_tracker = Sort(max_age=25, min_hits=4, iou_threshold=0.3)



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(
            QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(
            self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(
            self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(
            self.pushButton_video, 0, QtCore.Qt.AlignHCenter)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_3.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(
            self.pushButton_3, 0, QtCore.Qt.AlignHCenter)

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        self.pushButton_4.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_4.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(
            self.pushButton_4, 0, QtCore.Qt.AlignHCenter)

        

        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI防疫-人群距离监测与口罩识别系统"))
        self.pushButton_img.setText(_translate("MainWindow", "人群距离监测"))
        self.pushButton_camera.setText(_translate("MainWindow", "口罩识别"))
        self.pushButton_video.setText(_translate("MainWindow", "距离#口罩"))
        self.pushButton_3.setText(_translate("MainWindow", "区域热力图"))
        self.pushButton_4.setText(_translate("MainWindow", "退出系统"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_video_open)
        self.pushButton_video.clicked.connect(self.button_video1_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_3.clicked.connect(self.button_image_open)
        self.pushButton_4.clicked.connect(self.button_3_open)
        self.timer_video.timeout.connect(self.show_video_frame)


    def init_logo(self):
        pix = QtGui.QPixmap('./page.jpg')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        self._trackMap = np.zeros((360,640,3), dtype=np.uint8)#(1080,1920,3)
        self._crowdMap = np.zeros((360,640), dtype=np.int) 
        self.colorPool = ColorGenerator(size = 3000)
        self._centroid_dict = dict()
        self._numberOFpeople = list()
        self._greenZone = list()
        self._redZone = list()
        self._yellowZone = list()
        self._final_redZone = list()
        self._relation = dict()
        self._couples = dict()
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        print("-----+++",video_name)
        if not video_name:
            return
        self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
            *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        self.timer_video.start(30)
        self.pushButton_video.setDisabled(True)
        self.pushButton_img.setDisabled(True)
        self.pushButton_camera.setDisabled(True)
        dataset = LoadImages(video_name, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        print(dataset)
        bs = 1  # batch_size
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        dt_mask, seen_mask = [0.0, 0.0, 0.0], 0
        idnum = []
        risks=[]
        for path, im, im0s, vid_cap, s in dataset:
            # print(im.shape,im0s.shape)
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            pred = self.model(im, augment=self.opt.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                       self.opt.agnostic_nms, max_det=self.opt.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            numpre =[]
            numred = []
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                # print(im0.shape)
                # im0 = cv2.resize(im0,(960,540))
                im0 = cv2.resize(im0, (640, 360))
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop_trackMap
                imcc = im0.copy() if self.opt.save_crop else im0
                # print(imcc.shape)

                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    person_list = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        person = {0: "person"}
                        # if save_txt:  # Write to file
                        # print(xyxy)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = ((int(cls), str(float(conf)), (*xywh,)))  # label format
                        person_list.append(line)

                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # c = int(cls)  # integer class
                        # label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(person_list,type(person_list[0][2][0]))
                    humans = extract_humans(person_list)
                    # print(humans)
                    track_bbs_ids = self.mot_tracker.update(humans) if len(humans) != 0 else humans
                    # print(track_bbs_ids)

                    self._centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, imcc, self.calibration,
                                                                        self._centroid_dict, self.CorrectionShift,
                                                                        self.HumanHeightLimit)
                    redZone, greenZone = find_zone(centroid_dict, self._greenZone, self._redZone,
                                                   criteria=self.ViolationDistForIndivisuals)
                    if self.CouplesDetection:

                        e = birds_eye(imcc, self.calibration)
                        self._relation, relation = find_relation(e, centroid_dict, self.MembershipDistForCouples, redZone,
                                                            self._couples, self._relation)
                        self._couples, couples, coupleZone = find_couples(imcc, self._centroid_dict, relation,
                                                                     self.MembershipTimeForCouples, self._couples)
                        # print(_couples)
                        yellowZone, final_redZone, redGroups = find_redGroups(imcc, centroid_dict, self.calibration,
                                                                              self.ViolationDistForCouples, redZone,
                                                                              coupleZone, couples, self._yellowZone,
                                                                              self._final_redZone)
                    else:
                        couples = []
                        coupleZone = []
                        yellowZone = []
                        redGroups = redZone
                        final_redZone = redZone
                    
                    SDimage, birdSDimage, find_red = Apply_ellipticBound(centroid_dict, imcc, self.calibration, redZone,
                                                                         greenZone, yellowZone, final_redZone,
                                                                         coupleZone, couples,
                                                                         self.CircleradiusForIndivsual,
                                                                         self.CircleradiusForCouples)
                    # print(image.shape)
                    # cv2.rectangle(birdSDimage,(170,0),(360,300),(0,0,255),2)
                    self._crowdMap, crowdMap = Apply_crowdMap(centroid_dict, imcc, self._crowdMap)
                    crowd = (crowdMap - crowdMap.min()) / (crowdMap.max() - crowdMap.min())*255
                    crowd_visualShow, crowd_visualBird, crowd_histMap = VisualiseResult(crowd, e)
                    for id, box in centroid_dict.items():
                        idnum.append(id)
                        numpre.append(id)
                    idnum0 = idnum
                    res = list(set(idnum0))
                    risk = len(final_redZone)/(len(numpre)+0.000000000001)
                    risks.append(risk)
                    ri = np.mean(risks)
                    part = birdSDimage[0:300, 170:360, :]
                    part = cv2.resize(part, (120, 360))

                    # print("fps= %.2f"%(fps))
                    # SDimage = cv2.addWeighted(DTC_image,1, SDimage,1,10)
                
                    fps = (1. / (time_sync() - t1))
                    crowd_visualShow = cv2.resize(crowd_visualShow, (520, 360))
                    crowd_visualShow = cv2.putText(crowd_visualShow, "FPS: %.2f" % (fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    crowd_visualShow = cv2.putText(crowd_visualShow, "People Counting:" +str(len(res)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    crowd_visualShow = cv2.putText(crowd_visualShow, "Risk Evaluation: %.2f" % (ri), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)  
                    imgStackH = np.hstack((crowd_visualShow, part))
                    if find_red == 1:
                        imgStackH = cv2.putText(imgStackH, " Warning!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 0, 255), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 0, 255), 2)
                    else:
                        imgStackH = imgStackH
                        imgStackH = cv2.putText(imgStackH, " Normal!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 255, 0), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 255, 0), 2)
                    # cv2.imshow("imgStackH", imgStackH)
                    cv2.waitKey(1)

                    self.out.write(imgStackH)
                    show = cv2.resize(imgStackH, (640, 480))
                    self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                             QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
    def button_video1_open(self):
        self._trackMap = np.zeros((360,640,3), dtype=np.uint8)#(1080,1920,3)
        self._crowdMap = np.zeros((360,640), dtype=np.int) 
        self.colorPool = ColorGenerator(size = 3000)
        self._centroid_dict = dict()
        self._numberOFpeople = list()
        self._greenZone = list()
        self._redZone = list()
        self._yellowZone = list()
        self._final_redZone = list()
        self._relation = dict()
        self._couples = dict()
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        print("-----+++",video_name)
        if not video_name:
            return
        self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
            *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        self.timer_video.start(30)
        self.pushButton_video.setDisabled(True)
        self.pushButton_img.setDisabled(True)
        self.pushButton_camera.setDisabled(True)
        dataset = LoadImages(video_name, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        print(dataset)
        bs = 1  # batch_size
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        if self.opt.Mask:
            self.model_mask.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        dt_mask, seen_mask = [0.0, 0.0, 0.0], 0
        idnum = []
        risks=[]
        for path, im, im0s, vid_cap, s in dataset:
            # print(im.shape,im0s.shape)
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            pred = self.model(im, augment=self.opt.augment, visualize=visualize)
            if self.opt.Mask:
                pred_mask = self.model_mask(im, augment=self.opt.augment, visualize=self.opt.visualize)  #
                self.pred_mask = non_max_suppression(pred_mask, 0.6, self.opt.iou_thres,
                                                     self.opt.classes, self.opt.agnostic_nms,
                                                     max_det=self.opt.max_det)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                       self.opt.agnostic_nms, max_det=self.opt.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            numpre =[]
            numred = []
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                # print(im0.shape)
                # im0 = cv2.resize(im0,(960,540))
                im0 = cv2.resize(im0, (640, 360))
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop_trackMap
                imcc = im0.copy() if self.opt.save_crop else im0
                # print(imcc.shape)

                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    person_list = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        person = {0: "person"}
                        # if save_txt:  # Write to file
                        # print(xyxy)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = ((int(cls), str(float(conf)), (*xywh,)))  # label format
                        person_list.append(line)

                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # c = int(cls)  # integer class
                        # label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(person_list,type(person_list[0][2][0]))
                    humans = extract_humans(person_list)
                    # print(humans)
                    track_bbs_ids = self.mot_tracker.update(humans) if len(humans) != 0 else humans
                    # print(track_bbs_ids)

                    self._centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, imcc, self.calibration,
                                                                        self._centroid_dict, self.CorrectionShift,
                                                                        self.HumanHeightLimit)
                    redZone, greenZone = find_zone(centroid_dict, self._greenZone, self._redZone,
                                                   criteria=self.ViolationDistForIndivisuals)
                    if self.CouplesDetection:

                        e = birds_eye(imcc, self.calibration)
                        self._relation, relation = find_relation(e, centroid_dict, self.MembershipDistForCouples, redZone,
                                                            self._couples, self._relation)
                        self._couples, couples, coupleZone = find_couples(imcc, self._centroid_dict, relation,
                                                                     self.MembershipTimeForCouples, self._couples)
                        # print(_couples)
                        yellowZone, final_redZone, redGroups = find_redGroups(imcc, centroid_dict, self.calibration,
                                                                              self.ViolationDistForCouples, redZone,
                                                                              coupleZone, couples, self._yellowZone,
                                                                              self._final_redZone)
                    else:
                        couples = []
                        coupleZone = []
                        yellowZone = []
                        redGroups = redZone
                        final_redZone = redZone
                    if self.opt.Mask:
                        for i_mask, det_mask in enumerate(self.pred_mask):  # per image
                            seen_mask += 1

                            p_mask, im0_mask, frame_mask = path, im0s.copy(), getattr(dataset, 'frame', 0)

                            # print(im0_mask.shape)
                            im0_mask = cv2.resize(im0_mask, (640, 360))
                            annotator_mask = Annotator(im0_mask, line_width=self.opt.line_thickness,
                                                       example=str(self.names_mask))
                            if len(det_mask):
                                # Rescale boxes from img_size to im0 size
                                det_mask[:, :4] = scale_coords(im.shape[2:], det_mask[:, :4],
                                                               im0_mask.shape).round()

                                # Print results
                                for c_mask in det_mask[:, -1].unique():
                                    n_mask = (det_mask[:, -1] == c_mask).sum()  # detections per class
                                # Write results
                                for *xyxy_mask, conf_mask, cls_mask in reversed(det_mask):
                                    c_mask = int(cls_mask)  # integer class
                                    label_mask = None if self.opt.hide_labels else (
                                        self.names_mask[
                                            c_mask] if self.opt.hide_conf else f'{self.names_mask[c_mask]} {conf_mask:.2f}')
                                    # print(label_mask)
                                    annotator_mask.box_label(xyxy_mask, label_mask, color=colors(c_mask, True))
                    SDimage, birdSDimage, find_red = Apply_ellipticBound(centroid_dict, imcc, self.calibration, redZone,
                                                                         greenZone, yellowZone, final_redZone,
                                                                         coupleZone, couples,
                                                                         self.CircleradiusForIndivsual,
                                                                         self.CircleradiusForCouples)
                    # print(image.shape)
                    # cv2.rectangle(birdSDimage,(170,0),(360,300),(0,0,255),2)
                    for id, box in centroid_dict.items():
                        idnum.append(id)
                        numpre.append(id)
                    idnum0 = idnum
                    res = list(set(idnum0))
                    risk = len(final_redZone)/(len(numpre)+0.000000000001)
                    risks.append(risk)
                    ri = np.mean(risks)
                    part = birdSDimage[0:300, 170:360, :]
                    part = cv2.resize(part, (120, 360))
                    #轨迹跟踪
                    #self._trackMap = Apply_trackmap(centroid_dict, self._trackMap, self.colorPool, 3)
                    #SDimage = cv2.add(e.convrt2Image(self._trackMap), SDimage)

                    # print("fps= %.2f"%(fps))
                    # SDimage = cv2.addWeighted(DTC_image,1, SDimage,1,10)
                    if self.opt.Mask:
                        MASK = annotator_mask.result()
                        SDimage = cv2.addWeighted(SDimage, .5, MASK, .5, 10)
                    fps = (1. / (time_sync() - t1))
                    SDimage = cv2.resize(SDimage, (520, 360))
                    SDimage = cv2.putText(SDimage, "FPS: %.2f" % (fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    SDimage = cv2.putText(SDimage, "People Counting:"+str(len(res)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    SDimage = cv2.putText(SDimage, "Risk Evaluation: %.2f" % (ri), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)  
                    imgStackH = np.hstack((SDimage, part))
                    if find_red == 1:
                        imgStackH = cv2.putText(imgStackH, " Warning!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 0, 255), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 0, 255), 2)
                    else:
                        imgStackH = imgStackH
                        imgStackH = cv2.putText(imgStackH, " Normal!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 255, 0), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 255, 0), 2)
                    # cv2.imshow("imgStackH", imgStackH)
                    cv2.waitKey(1)

                    self.out.write(imgStackH)
                    show = cv2.resize(imgStackH, (640, 480))
                    self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                             QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
    def button_video_open(self):
        self._trackMap = np.zeros((360,640,3), dtype=np.uint8)#(1080,1920,3)
        self._crowdMap = np.zeros((360,640), dtype=np.int) 
        self.colorPool = ColorGenerator(size = 3000)
        self._centroid_dict = dict()
        self._numberOFpeople = list()
        self._greenZone = list()
        self._redZone = list()
        self._yellowZone = list()
        self._final_redZone = list()
        self._relation = dict()
        self._couples = dict()
        Mask1 = 0
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        print("-----+++",video_name)
        if not video_name:
            return
        self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
            *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        self.timer_video.start(30)
        self.pushButton_video.setDisabled(True)
        self.pushButton_img.setDisabled(True)
        self.pushButton_camera.setDisabled(True)
        dataset = LoadImages(video_name, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        print(dataset)
        bs = 1  # batch_size
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        if Mask1:
            self.model_mask.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        dt_mask, seen_mask = [0.0, 0.0, 0.0], 0
        idnum = []
        risks=[]
        for path, im, im0s, vid_cap, s in dataset:
            # print(im.shape,im0s.shape)
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            pred = self.model(im, augment=self.opt.augment, visualize=visualize)
            if Mask1:
                pred_mask = self.model_mask(im, augment=self.opt.augment, visualize=self.opt.visualize)  #
                self.pred_mask = non_max_suppression(pred_mask, self.opt.conf_thres, self.opt.iou_thres,
                                                     self.opt.classes, self.opt.agnostic_nms,
                                                     max_det=self.opt.max_det)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                       self.opt.agnostic_nms, max_det=self.opt.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            numpre =[]
            numred = []
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                # print(im0.shape)
                # im0 = cv2.resize(im0,(960,540))
                im0 = cv2.resize(im0, (640, 360))
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop_trackMap
                imcc = im0.copy() if self.opt.save_crop else im0
                # print(imcc.shape)

                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    person_list = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        person = {0: "person"}
                        # if save_txt:  # Write to file
                        # print(xyxy)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = ((int(cls), str(float(conf)), (*xywh,)))  # label format
                        person_list.append(line)

                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # c = int(cls)  # integer class
                        # label = None if self.opt.hide_labels else (self.names[c] if self.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(person_list,type(person_list[0][2][0]))
                    humans = extract_humans(person_list)
                    # print(humans)
                    track_bbs_ids = self.mot_tracker.update(humans) if len(humans) != 0 else humans
                    # print(track_bbs_ids)

                    self._centroid_dict, centroid_dict, partImage = centroid(track_bbs_ids, imcc, self.calibration,
                                                                        self._centroid_dict, self.CorrectionShift,
                                                                        self.HumanHeightLimit)
                    redZone, greenZone = find_zone(centroid_dict, self._greenZone, self._redZone,
                                                   criteria=self.ViolationDistForIndivisuals)
                    if self.CouplesDetection:

                        e = birds_eye(imcc, self.calibration)
                        self._relation, relation = find_relation(e, centroid_dict, self.MembershipDistForCouples, redZone,
                                                            self._couples, self._relation)
                        self._couples, couples, coupleZone = find_couples(imcc, self._centroid_dict, relation,
                                                                     self.MembershipTimeForCouples, self._couples)
                        # print(_couples)
                        yellowZone, final_redZone, redGroups = find_redGroups(imcc, centroid_dict, self.calibration,
                                                                              self.ViolationDistForCouples, redZone,
                                                                              coupleZone, couples, self._yellowZone,
                                                                              self._final_redZone)
                    else:
                        couples = []
                        coupleZone = []
                        yellowZone = []
                        redGroups = redZone
                        final_redZone = redZone
                    if Mask1:
                        for i_mask, det_mask in enumerate(self.pred_mask):  # per image
                            seen_mask += 1

                            p_mask, im0_mask, frame_mask = path, im0s.copy(), getattr(dataset, 'frame', 0)

                            # print(im0_mask.shape)
                            im0_mask = cv2.resize(im0_mask, (640, 360))
                            annotator_mask = Annotator(im0_mask, line_width=self.opt.line_thickness,
                                                       example=str(self.names_mask))
                            if len(det_mask):
                                # Rescale boxes from img_size to im0 size
                                det_mask[:, :4] = scale_coords(im.shape[2:], det_mask[:, :4],
                                                               im0_mask.shape).round()

                                # Print results
                                for c_mask in det_mask[:, -1].unique():
                                    n_mask = (det_mask[:, -1] == c_mask).sum()  # detections per class
                                # Write results
                                for *xyxy_mask, conf_mask, cls_mask in reversed(det_mask):
                                    c_mask = int(cls_mask)  # integer class
                                    label_mask = None if self.opt.hide_labels else (
                                        self.names_mask[
                                            c_mask] if self.opt.hide_conf else f'{self.names_mask[c_mask]} {conf_mask:.2f}')
                                    # print(label_mask)
                                    annotator_mask.box_label(xyxy_mask, label_mask, color=colors(c_mask, True))
                    SDimage, birdSDimage, find_red = Apply_ellipticBound(centroid_dict, imcc, self.calibration, redZone,
                                                                         greenZone, yellowZone, final_redZone,
                                                                         coupleZone, couples,
                                                                         self.CircleradiusForIndivsual,
                                                                         self.CircleradiusForCouples)
                    # print(image.shape)
                    # cv2.rectangle(birdSDimage,(170,0),(360,300),(0,0,255),2)
                    
                    for id, box in centroid_dict.items():
                        idnum.append(id)
                        numpre.append(id)
                    idnum0 = idnum
                    res = list(set(idnum0))
                    risk = len(final_redZone)/(len(numpre)+0.000000000001)
                    risks.append(risk)
                    ri = np.mean(risks)
                    part = birdSDimage[0:300, 170:360, :]
                    part = cv2.resize(part, (120, 360))
                    self._trackMap = Apply_trackmap(centroid_dict, self._trackMap, self.colorPool, 3)
                    SDimage = cv2.add(e.convrt2Image(self._trackMap), SDimage)

                    # print("fps= %.2f"%(fps))
                    # SDimage = cv2.addWeighted(DTC_image,1, SDimage,1,10)
                    if Mask1:
                        MASK = annotator_mask.result()
                        SDimage = cv2.addWeighted(SDimage, .5, MASK, .5, 10)
                    fps = (1. / (time_sync() - t1))
                    SDimage = cv2.resize(SDimage, (520, 360))
                    SDimage = cv2.putText(SDimage, "FPS: %.2f" % (fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    SDimage = cv2.putText(SDimage, "People Counting:" +str(len(res)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)
                    SDimage = cv2.putText(SDimage, "Risk Evaluation: %.2f" % (ri), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .4,
                                          (0, 255, 0), 1)  
                    imgStackH = np.hstack((SDimage, part))
                    if find_red == 1:
                        imgStackH = cv2.putText(imgStackH, " Warning!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 0, 255), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 0, 255), 2)
                    else:
                        imgStackH = imgStackH
                        imgStackH = cv2.putText(imgStackH, " Normal!", (520, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                                (0, 255, 0), 1)
                        cv2.rectangle(imgStackH, (520, 0), (640, 360), (0, 255, 0), 2)
                    # cv2.imshow("imgStackH", imgStackH)
                    cv2.waitKey(1)

                    self.out.write(imgStackH)
                    show = cv2.resize(imgStackH, (640, 480))
                    self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                             QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
    def button_camera_open(self):

        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:

                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setDisabled(True)

                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(str(0), img_size=self.imgsz, stride=self.stride, auto=self.pt)
                bs = len(dataset)
                self.model_mask.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
                dt_mask, seen_mask = [0.0, 0.0, 0.0], 0
                for path, im, im0s, vid_cap, s in dataset:
                    # print(im.shape,im0s.shape)
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(self.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    pred_mask = self.model_mask(im, augment=self.opt.augment, visualize=self.opt.visualize)  #
                    self.pred_mask = non_max_suppression(pred_mask, self.opt.conf_thres, self.opt.iou_thres,
                                                         self.opt.classes, self.opt.agnostic_nms,
                                                         max_det=self.opt.max_det)



                    for i_mask, det_mask in enumerate(self.pred_mask):  # per image
                        seen_mask += 1
                        p_mask, im0_mask, frame_mask = path[i_mask], im0s[i_mask].copy(), dataset.count

                        im0_mask = cv2.resize(im0_mask, (640, 360))
                        annotator_mask = Annotator(im0_mask, line_width=self.opt.line_thickness,
                                                   example=str(self.names_mask))
                        if len(det_mask):
                            # Rescale boxes from img_size to im0 size
                            det_mask[:, :4] = scale_coords(im.shape[2:], det_mask[:, :4],
                                                           im0_mask.shape).round()

                            # Print results
                            for c_mask in det_mask[:, -1].unique():
                                n_mask = (det_mask[:, -1] == c_mask).sum()  # detections per class
                            # Write results
                            for *xyxy_mask, conf_mask, cls_mask in reversed(det_mask):
                                c_mask = int(cls_mask)  # integer class
                                label_mask = None if self.opt.hide_labels else (
                                    self.names_mask[
                                        c_mask] if self.opt.hide_conf else f'{self.names_mask[c_mask]} {conf_mask:.2f}')
                                # print(label_mask)
                                annotator_mask.box_label(xyxy_mask, label_mask, color=colors(c_mask, True))
                        MASK = annotator_mask.result()
                        fps = (1. / (time_sync() - t1))
                        MASK = cv2.putText(MASK, "FPS: %.2f" % (fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                              (0, 255, 0), 1)
                        cv2.waitKey(1)

                        self.out.write(MASK)
                        show = cv2.resize(MASK, (640, 480))
                        self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                                 QtGui.QImage.Format_RGB888)
                        self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")
    def button_3_open(self):
        #return QCoreApplication.quit()
        return sys.exit()


    def show_video_frame(self):

        self.timer_video.stop()
        self.cap.release()
        self.out.release()
        self.label.clear()
        self.pushButton_video.setDisabled(False)
        self.pushButton_img.setDisabled(False)
        self.pushButton_camera.setDisabled(False)
        self.init_logo()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
