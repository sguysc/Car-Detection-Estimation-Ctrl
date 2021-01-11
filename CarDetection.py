#!/usr/bin/env python
# coding: utf-8
"""
Object detection from a video and tracking for autonomous car
=============================================================

GUY
11/01/2021
"""

# %% imports stuff
# general stuff
import numpy as np
import os
from pathlib import Path
import datetime
from pytube import YouTube
from copy import copy
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw, ImageFont
from stonesoup.reader.video import VideoClipReader

# estimator stuff
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, RandomWalk)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.detector.base import Detector
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.detection import Detection
from stonesoup.base import Property
from stonesoup.feeder.filter import MetadataValueFilter

# YOLO stuff
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

# helper function to get a video from youtube to analyze
def LoadVideoFromYouTube(t0, tf, VIDEO_FILENAME = 'sample1', \
                         URL='http://www.youtube.com/watch?v=MNn9qKG2UFI'):
    VIDEO_EXTENTION = '.mp4'
    VIDEO_PATH = os.path.join(os.getcwd(), VIDEO_FILENAME+VIDEO_EXTENTION)
    
    if not os.path.exists(VIDEO_PATH):
        yt = YouTube(URL)
        yt.streams[0].download(filename=VIDEO_FILENAME)
    
    # clip time of video from t0 to tf
    start_time = datetime.timedelta(minutes=0, seconds=t0)
    end_time   = datetime.timedelta(minutes=0, seconds=tf)
    frame_reader = VideoClipReader(VIDEO_PATH, start_time, end_time)
    
    # can also clip the movie if we want    
    # frame_reader.clip = all.crop(frame_reader.clip, 100, 100)
    # num_frames = len(list(frame_reader.clip.iter_frames()))

    return frame_reader

# display the video
def ShowVideo(frame_reader):
    # fig, ax = plt.subplots(num="VideoClipReader output")
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    artists = []
    num_frames = len(list(frame_reader.clip.iter_frames()))
    
    print('Running FrameReader example...')
    for timestamp, frame in frame_reader:
        if not (len(artists)+1) % 10:
            print("Frame: {}/{}".format(len(artists)+1, num_frames))
    
        # Read the frame pixels
        pixels = copy(frame.pixels)
    
        # Plot output
        image = Image.fromarray(pixels)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        fig.tight_layout()
        artist = ax.imshow(image, animated=True)
        artists.append([artist])
    
    ani = animation.ArtistAnimation(fig, artists, interval=100, blit=True, repeat_delay=200)
    plt.show()


# visualising detections, [x, y, w, h] - all the bounding boxes of relevant objects
# with relevant probability score
def draw_detections(image, detections, show_class=False, show_score=False):
    # drawing them on the picture
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x0, y0, w, h = np.array(detection.state_vector).reshape(4)
        x1, y1 = (x0 + w, y0 + h)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=1)
        class_ = detection.metadata['class']['name']
        score = round(float(detection.metadata['score']),2)
        if show_class and show_score:
            draw.text((x0,y1 + 2), '{}:{}'.format(class_, score), fill=(0, 255, 0))
        elif show_class:
            draw.text((x0, y1 + 2), '{}'.format(class_), fill=(0, 255, 0))
        elif show_score:
            draw.text((x0, y1 + 2), '{}'.format(score), fill=(0, 255, 0))

    return image

# visualising tracks - all the objects boxes that are active for at least some time 
# defined by the tracker
def draw_tracks(image, tracks, show_history=True, show_class=True, show_score=True):
    draw = ImageDraw.Draw(image)
    for track in tracks:
        bboxes = np.array([np.array(state.state_vector[[0, 2, 4, 5]]).reshape(4)
                           for state in track.states])
        x0, y0, w, h = bboxes[-1]
        x1 = x0 + w
        y1 = y0 + h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        if show_history:
            pts = [(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in bboxes]
            draw.line(pts, fill=(255, 0, 0), width=2)

        class_ = track.metadata['class']['name']
        score = round(float(track.metadata['score']), 2)
        if show_class and show_score:
            draw.text((x0, y1 + 2), '{}:{}'.format(class_, score), fill=(255, 0, 0), font=fnt)
        elif show_class:
            draw.text((x0, y1 + 2), '{}'.format(class_), fill=(255, 0, 0))
        elif show_score:
            draw.text((x0, y1 + 2), '{}'.format(score), fill=(255, 0, 0))
    return image

# helper class that creates the detections. implements a Detector class.
# this detector uses Yolov5 as its object detector
class YOLOv5Detector(Detector):
    
    model_path: Path = Property(
        doc="Path to ``saved_model`` directory. This is the directory that contains the "
            "``saved_model.pb`` file.")

    labels_path: Path = Property(
        doc="Path to label map (``*.pbtxt`` file). This is the file that contains mapping of "
            "object/class ids to meaningful names")

    run_async: bool = Property(
        doc="If set to ``True``, the detector will digest frames from the reader asynchronously "
            "and only perform detection on the last frame digested. This is suitable when the "
            "detector is applied to readers generating a live feed (e.g. "
            ":class:`~.FFmpegVideoStreamReader`), where real-time processing is paramount. "
            "Defaults to ``False``",
        default=False)

    def __init__(self, *args, **kwargs):
        # print('in YOLOv5Detector.__init__')
        super().__init__(*args, **kwargs)
        # classes to detect
        classes = [0, 2, 5, 7]
        
        self.debug =  True #False
        self.yolo_detections = {}
        self.main_dir  = os.getcwd() +   '/'

        self.weights  = [*args][1] + 'yolov5s.pt'
        self.source   = [*args][1] + 'KITTI/testing/images/2011_09_26_drive_0059_*.png'
        self.view_img = False
        self.save_txt = False 
        self.imgsz    = 640
        self.augment  = True
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = True
        self.agnostic = True
        # all available classes
        self.category_index = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', \
                   4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}  # classes should be ints.
        self.classes = classes

        self.device = select_device('cpu') # that's what I have :|

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size

        # Set Dataloader
        self.dataset = LoadImages(self.source, img_size=self.imgsz)
        self.num_frames = len(self.dataset)
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        
        self.path = ''
        self.img  = None
        self.im0s = None
    
    @BufferedGenerator.generator_method
    def detections_gen(self):
        """Returns a generator of detections for each frame.

        Yields datetime.datetime, Detection
        Detections generated in the time step. The detection state vector is of the form
        (x, y, w, h), where ``x, y`` denote the relative coordinates of the top-left
        corner of the bounding box containing the object, while ``w, h`` denote the relative
        width and height of the bounding box. Additionally, each detection carries the
        following meta-data fields:

        - ``raw_box``: The raw bounding box, as generated by TensorFlow.
        - ``class``: A dict with keys ``id`` and ``name`` relating to the \
              id and name of the detection class, as specified by the label map.
        - ``score``: A float in the range ``(0, 1]`` indicating the detector's confidence
        """
        yield from self._detections_gen()
        
    def _capture(self):
        pass
    
    def _detections_gen_async(self):
        return self._detections_gen()
    
    def _detections_gen(self):
        # prepare the next picture to analyze, and output all the detections
        frame = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        
        for path, frame, im0s, vid_cap in self.dataset:
            img = copy(frame)
            frame = torch.from_numpy(frame).to(self.device)
            frame = frame.float()  # uint8 to fp16/32
            frame /= 255.0  # 0 - 255 to 0.0 - 1.0
            if frame.ndimension() == 3:
                frame = frame.unsqueeze(0)
            
            self.timestamp = datetime.datetime.today()
            detections = self._get_detections_from_frame(frame, im0s)
            self.current = copy(detections)
            # save for later access
            self.path, self.img, self.im0s = path, img, im0s
            yield self.timestamp, detections

    def _get_detections_from_frame(self, frame, im0):
        # where the YOLOv5 does the heavy lifting
        pred = self.model(frame, augment=self.augment)[0]
    
        # Apply NMS 
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, \
                                   classes=self.classes, agnostic=self.agnostic_nms)

        # Process detections
        # Extract classes, boxes and scores
        classes = []
        boxes = []
        scores = []
        
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # normalized xywh
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                    # classes.append(det[i, -1])
                    # boxes.append([xywh[0], xywh[1], xywh[2], xywh[3]])
                    boxes.append([ int(xyxy[0]), int(xyxy[1]), \
                                   int(xyxy[2]), int(xyxy[3]) ])
                    #import pdb; pdb.set_trace()
                    scores.append( conf )
                    classes.append(np.int(cls))

        # num_detections = int(len(scores))

        # Form detections and metadata
        detections = set()
        frame_height, frame_width, _ = im0.shape
        for box, class_, score in zip(boxes, classes, scores):
            metadata = {
                "raw_box": box,
                "class": {'name': self.category_index[class_]},
                "score": score
            }
            # Transform box to be in format (x, y, w, h)
            state_vector = StateVector([box[0],box[1],box[2]-box[0],box[3]-box[1]])
            # [box[1]*frame_width, box[0]*frame_height, (box[3] - box[1])*frame_width, (box[2] - box[0])*frame_height]
            detection = Detection(state_vector=state_vector,
                                  timestamp=self.timestamp,
                                  metadata=metadata)
            detections.add(detection)

        return detections

# a Multi-Object Video Tracker
class MultiObjectVideoTracker(object):
    def __init__(self):
        # state [x_k, xdot_k, y_k, ydot_k, w_k, h_k]
        # x_k, y_k move with constant velocity and white noise acceleration.
        # width and height of the bounding box is a random walk
        # x_k, y_k are the top-left corners of the bounding box
        self.t_models = [ConstantVelocity(20**2), ConstantVelocity(20**2), \
                         RandomWalk(20**2), RandomWalk(20**2)]
        self.transition_model = CombinedLinearGaussianTransitionModel(self.t_models)

        # Measurement Model:  z_k = [x_k, y_k, w_k, h_k]
        # map to indices `[0,2,4,5]` of the 6-dimensional state 
        self.measurement_model = LinearGaussian(ndim_state=6, mapping=[0, 2, 4, 5],
                                   noise_covar=np.diag([1**2, 1**2, 3**2, 3**2]))
        
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater   = KalmanUpdater(self.measurement_model)
        
        # Data Association, generate hypotheses between tracks and
        # measurements, where Mahalanobis distance is used as a measure of quality:
        self.hypothesiser = DistanceHypothesiser(self.predictor, self.updater, Mahalanobis(), 10)
        
        # perform fast joint data association, based on the Global Nearest Neighbour (GNN) algorithm:
        self.data_associator = GNNWith2DAssignment(self.hypothesiser)

        # Track Initiation: tentatively initiate tracks from unassociated measurements, and hold them within the
        # initiator until they have survived for at least 10 frames. We also define a
        # deleter to be used by the initiator to delete tentative tracks
        # that have not been associated to a measurement in the last 3 frames.
        prior_state = GaussianState(StateVector(np.zeros((6,1))),
                                    CovarianceMatrix(np.diag([100**2, 30**2, 100**2, 30**2, 100**2, 100**2])))
        deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
        self.initiator = MultiMeasurementInitiator(prior_state, self.measurement_model, deleter_init,
                                              self.data_associator, self.updater, min_points=10)

        # Track Deletion, for confirmed tracks we delete tracks after they have not been
        # associated to a measurement in the last 15 frames.
        deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)
        
        # the detector module, YOLOv5
        # 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        # 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
        self._yolo_detector = YOLOv5Detector(None, './', './',
                                       run_async=False)
        self.detector = MetadataValueFilter(self._yolo_detector, \
                                    'class', lambda x: x['name'] in ['car', 'bus', 'truck'])
        self.detector = MetadataValueFilter(self.detector, \
                                    'score', lambda x: x > 0.7)
        
        # Building the tracker
        self.tracker = MultiTargetTracker(
            initiator=self.initiator,
            deleter=deleter,
            detector=self.detector,
            data_associator=self.data_associator,
            updater=self.updater,
        )

    def run(self):
        fig3, ax3 = plt.subplots(num="MultiTargetTracker output")
        fig3.tight_layout()
        artists3 = []
        num_frames = self._yolo_detector.num_frames
      
        # import pdb; pdb.set_trace()
        for timestamp, tracks in self.tracker:
            if not (len(artists3) + 1) % 10:
                print("Frame: {}/{}".format(len(artists3) + 1, num_frames))
        
            # Read the detections (index0 is timestep, index1 are the detections)
            det  = self.tracker.detector.current[1]
            im0s = self._yolo_detector.im0s        
            # Read frame
            pixels = copy(im0s) #copy(frame.pixels)
            # convert from CV2's BGR to RGB
            im_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

            # Plot output
            image = Image.fromarray(im_rgb)
            image = draw_detections(image, det)
            image = draw_tracks(image, tracks)
            ax3.axes.xaxis.set_visible(False)
            ax3.axes.yaxis.set_visible(False)
            fig3.tight_layout()
            artist = ax3.imshow(image, animated=True)
            artists3.append([artist])
        #import pdb; pdb.set_trace()
        ani3 = animation.ArtistAnimation(fig3, artists3, interval=200, \
                                  blit=True, repeat_delay=200)
        plt.pause(0.1)
        progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
        ani3.save('tracker_output.mp4', progress_callback=progress_callback)
        

if __name__ == '__main__':
    MOVT = MultiObjectVideoTracker()
    MOVT.run()
    print('done')
