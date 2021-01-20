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
import time 
from pytube import YouTube
from copy import copy
# import matplotlib 
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw, ImageFont
from stonesoup.reader.video import VideoClipReader
from MyPlotter import MyPlotter

# estimator stuff
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, RandomWalk)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import PDA
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.initiator.simple import SimpleMeasurementInitiator, MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import SingleTargetTracker, MultiTargetTracker
from stonesoup.detector.base import Detector
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.detection import Detection, Clutter
from stonesoup.base import Property
from stonesoup.feeder.filter import MetadataValueFilter
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors  # For storing state vectors during association
from stonesoup.functions import gm_reduce_single  # For merging states to get posterior estimate
from stonesoup.types.update import GaussianStateUpdate  # To store posterior estimate

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

MOVT = None
# plotter.ax.set_ylim(0, 25)

# in case we do not want to delete a track
class MyDeleter:
    def delete_tracks(self, tracks):
        return set()

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
    
        # if(len(artists)+1 > 1000):
        #     break
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
def draw_tracks(image, tracks, t, show_history=True, show_class=True, show_score=True, null_hyp=False):
    draw = ImageDraw.Draw(image)
    for track in tracks:
        try:
            bboxes = np.array([np.array(state.state_vector[[0, 2, 4, 5]]).reshape(4)
                           for state in track.states])
        except:
            # import pdb; pdb.set_trace()
            # this is for PDA, it has only one track
            bboxes = np.array([np.array(track.state_vector[[0,2,4,5]]).reshape(4)])
            
        x0, y0, w, h = bboxes[-1]
        x1 = x0 + w
        y1 = y0 + h
        line_color = (255, 0, 0)
        if(null_hyp):
            # got no detections, using null hyothesis + prediction, so changing color
            # import pdb; pdb.set_trace()
            line_color = (0, 0, 255)
            
        draw.rectangle([x0, y0, x1, y1], outline=line_color, width=2)
            
        if show_history:
            pts = [(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in bboxes]
            draw.line(pts, fill=line_color, width=2)

        try:
            class_ = track.metadata['class']['name']
            score = round(float(track.metadata['score']), 2)
        except:
            # again, for the PDA case
            class_ = 'car'
            # GUY: same problem, since no-detection out-weighs everything else, 
            # i'm currently ignoring that and watching between everything else.
            prob = np.array([np.float(i.probability) for i in track.hypothesis.single_hypotheses])
            if(len(prob) > 1):
                # then it has the null hypothesis in index 0, otherwise
                # it only has the null hypothesis so don't "re-normalize"
                # prob = prob[1:]/prob[1:].sum()
                # after changing the spatial_ratio_factor, the null hypothesis is valid
                prob = prob/prob.sum()
            # score = round(np.float(track.hypothesis.single_hypotheses[0].probability), 2)
            score = round(prob.max(), 2)
        
        if show_class and show_score:
            draw.text((x0, y1 + 2), '{}:{}'.format(class_, score), fill=line_color, font=fnt)
        elif show_class:
            draw.text((x0, y1 + 2), '{}'.format(class_), fill=line_color)
        elif show_score:
            draw.text((x0, y1 + 2), '{}'.format(score), fill=line_color)

    # import pdb; pdb.set_trace()
    text_x = image.size[0] / 2
    text_y = 10
    draw.text((text_x, text_y), 't=%d' %(t), fill=(128,128,128), font=fnt)
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
    
    classes: list = Property(doc='Which classes to detect', default=[0])

    def __init__(self, *args, **kwargs):
        # print('in YOLOv5Detector.__init__')
        super().__init__(*args, **kwargs)
        
        self.debug =  True #False
        self.yolo_detections = {}
        self.main_dir  = os.getcwd() +   '/'
        # import pdb; pdb.set_trace()
        self.weights  = [*args][1] + 'yolov5s.pt'
        # self.source   = [*args][1] + 'KITTI/testing/images/2011_09_26_drive_0059_*.png'
        # self.source   = '/home/cornell/Tools/yolov5/KITTI/training/images/2011_09_26_drive_0057_*.png'
        # self.source   = '/home/cornell/Tools/yolov5/KITTI/training/images/2011_09_26_drive_0059_*.png'
        # start_at, end_at= -1, -1
        self.source = './test1.mp4'
        # how many frames to take from the video
        start_at = 30*30 #54.5*60*30 
        end_at = 60*30 #55.5*60*30
        # start_at = 11*60*30 
        # end_at = (11*60+30)*30 
        # start_at = (46*60+20)*30 
        # end_at = (46*60+50)*30 
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
        self.classes = kwargs['classes']

        self.device = select_device('cpu') # that's what I have :|

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size

        # Set Dataloader
        self.dataset = LoadImages(self.source, img_size=self.imgsz, \
                                  start_at=start_at, end_at=end_at)
        try:
            # video file
            self.num_frames = int(self.dataset.nframes)
        except:
            # list of images
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
        
        # import pdb; pdb.set_trace()
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
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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
    def __init__(self, est_kind='Multi'):
        # store the tupe of filter
        self.est_kind = est_kind
        
        self.plotter = MyPlotter()
        # self.plotter.fig
        # time.sleep(1.0)
        
        # state [x_k, xdot_k, y_k, ydot_k, w_k, h_k]
        # x_k, y_k move with constant velocity and white noise acceleration.
        # width and height of the bounding box is a random walk
        # x_k, y_k are the top-left corners of the bounding box
        # self.t_models = [ConstantVelocity(20**2), ConstantVelocity(20**2), \
        #                  RandomWalk(20**2), RandomWalk(20**2)]
        self.t_models = [ConstantVelocity(5**2), ConstantVelocity(5**2), \
                         RandomWalk(5**2), RandomWalk(5**2)]
        self.transition_model = CombinedLinearGaussianTransitionModel(self.t_models)

        # Measurement Model:  z_k = [x_k, y_k, w_k, h_k]
        # map to indices `[0,2,4,5]` of the 6-dimensional state 
        self.measurement_model = LinearGaussian(ndim_state=6, mapping=[0, 2, 4, 5],
                                   noise_covar=np.diag([3**2, 3**2, 3**2, 3**2]))
        
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater   = KalmanUpdater(self.measurement_model)
        
        # Track Initiation: tentatively initiate tracks from unassociated measurements, and hold them within the
        # initiator until they have survived for at least 10 frames. We also define a
        # deleter to be used by the initiator to delete tentative tracks
        # that have not been associated to a measurement in the last 3 frames.
        # x0 = StateVector(np.zeros((6,1)))
        self.prior_state = GaussianState(
                        # StateVector(np.array([580., 0., 180., 0., 80., 80.]).reshape((6,1))),
                        # StateVector(np.array([580., 0., 180., 0., 200., 180.]).reshape((6,1))),
                        StateVector(np.array([300., 0., 200., 0., 40., 40.]).reshape((6,1))),
                        # StateVector(np.array([300., 0., 200., 0., 100., 80.]).reshape((6,1))),
                        CovarianceMatrix(np.diag([20**2, 10**2, 20**2, 10**2, 10**2, 10**2])),
                        timestamp = datetime.datetime.now())
        
        if(est_kind.upper() in ['MULTI']):
            # Data Association, generate hypotheses between tracks and
            # measurements, where Mahalanobis distance is used as a measure of quality:
            self.hypothesiser = DistanceHypothesiser(self.predictor, self.updater, Mahalanobis(), 10)
            # perform fast joint data association, based on the Global Nearest Neighbour (GNN) algorithm:
            self.data_associator = GNNWith2DAssignment(self.hypothesiser)
            deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
            # Track Deletion, for confirmed tracks we delete tracks after they have not been
            # associated to a measurement in the last 15 frames.
            deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)
            self.initiator = MultiMeasurementInitiator(
                                    prior_state=self.prior_state,
                                    measurement_model=self.measurement_model,
                                    deleter=deleter_init,
                                    data_associator=self.data_associator,
                                    updater=self.updater,
                                    min_points=10,
                                    )   
            
        elif(est_kind.upper() in ['PDA']):
            # PDA associator generate track predictions and calculate probabilities
            # for all prediction-detection pairs for a single prediction and multiple detections
            self.hypothesiser = PDAHypothesiser(predictor=self.predictor,
                                    updater=self.updater,
                                    clutter_spatial_density=1.0E-8,
                                    # clutter_spatial_density=1.0E-7,
                                    # clutter_spatial_density=0.125,
                                    prob_detect=0.9)
            # takes these hypotheses and returns a dictionary of key-value pairings of
            # each track and detection which it is to be associated with.
            self.data_associator = PDA(hypothesiser=self.hypothesiser)
            # deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
            # or,
            #covariance_limit_for_delete = 30
            #deleter_init = CovarianceBasedDeleter(covariance_limit_for_delete)
            # or
            deleter_init = MyDeleter()
            # Track Deletion, for confirmed tracks we delete tracks after they have not been
            # associated to a measurement in the last 15 frames.
            # deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)
            # or,
            #deleter = CovarianceBasedDeleter(covariance_limit_for_delete)
            # or,
            deleter = MyDeleter()
            
            self.initiator = SimpleMeasurementInitiator(prior_state=self.prior_state,
                                                        measurement_model=self.measurement_model)
                
        
        # the detector module, YOLOv5
        # 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        # 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
        self._yolo_detector = YOLOv5Detector(None, './', './',
                                       run_async=False, classes=[0, 2, 5, 7])
        self.detector = MetadataValueFilter(self._yolo_detector, \
                                    'class', lambda x: x['name'].lower() in ['car', 'bus', 'truck'])
        self.detector = MetadataValueFilter(self.detector, \
                                    'score', lambda x: x > 0.7)
        
                
        # Building the tracker
        if(est_kind.upper() in ['MULTI']):
            self.tracker = MultiTargetTracker(
                initiator=self.initiator,
                deleter=deleter,
                detector=self.detector,
                data_associator=self.data_associator,
                updater=self.updater,
            )
        elif(est_kind.upper() in ['PDA']):
            # self.tracker = SingleTargetTracker(
            #     initiator=self.initiator,
            #     deleter=deleter,
            #     detector=self.detector,
            #     data_associator=self.data_associator,
            #     updater=self.updater
            # )
            pass

    def run(self):
        if(self.est_kind.upper() in ['MULTI']):
            fig3, ax3 = plt.subplots(num="MultiTargetTracker output")
            fig3.tight_layout()
            artists3 = []
            num_frames = self._yolo_detector.num_frames
          
            # all_measurements = []
            # measurement_set = set()
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
            print("Frame: {}/{}".format(num_frames, num_frames))
            ani3 = animation.ArtistAnimation(fig3, artists3, interval=200, \
                                      blit=True, repeat_delay=200)
            plt.pause(0.1)
            print('Saving to mpeg ...')
            progress_callback = lambda i,n: print(f'Saving frame {i} of {num_frames}') if i % 10 == 0 else None
            ani3.save('tracker_output.mp4', progress_callback=progress_callback)
            
        elif(self.est_kind.upper() in ['PDA']):
            fig3, ax3 = plt.subplots(num="PDA output")
            fig3.tight_layout()
            artists3 = []
            num_frames = self._yolo_detector.num_frames
            self.trk = Track([self.prior_state])
            self.hyp_mean = []
            self.hyp_cov = []
            self.hyp_weight = []
            self.miss_det = []
            self.miss_det_mean = []
            self.miss_det_cov = []
            
            all_measurements = []            
            idx = 0
            # import pdb; pdb.set_trace()
            for n, measurements in self.detector:
                if not (idx+1) % 10:
                    print("Frame: {}/{}".format(idx+1, num_frames))
                
                # if(idx == 211):
                    # hypothesis changes abruptly
                    # import pdb; pdb.set_trace()
                    # break
                
                # import pdb; pdb.set_trace()
                # do the actual PDA algorithm
                hypotheses = self.data_associator.associate([self.trk],
                                                       measurements, n)
                
                hypotheses = hypotheses[self.trk]
            
                # Loop through each hypothesis, creating posterior states for each, and merge to calculate
                # approximation to actual posterior state mean and covariance.
                posterior_states = []
                posterior_state_weights = []
                only_prediction = False
                # import pdb; pdb.set_trace()
                for hypothesis in hypotheses:
                    if not hypothesis:
                        # this is the null hypothesis that we don't have a true detection.
                        # it is commented out at the moment because it out-weighs any of
                        # the true detections for some reason. GUY: check what is wrong
                        # with the parameters of the prior/transition/.. such that it 
                        # gets so much likelihood
                        posterior_states.append(hypothesis.prediction)
                        posterior_state_weights.append(hypothesis.probability)
                        if(hypothesis.probability > 0.90):
                            # too much probability it's a miss detection so the final
                            # track will be mostly prediction
                            only_prediction = True
                    else:
                    # if(hypothesis):
                        # these assume one of the measurement is correct:
                        posterior_state = self.updater.update(hypothesis)
                        posterior_states.append(posterior_state)
                        posterior_state_weights.append(hypothesis.probability)
            
                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)
                # No detections at this frame, so only the null hypothesis is valid
                if len(posterior_state_weights) == 1:
                    only_prediction = True
                    
                # import pdb; pdb.set_trace()
                max_ind = np.argmax(weights)
                self.hyp_mean.append(means[:,max_ind])
                self.hyp_cov.append(covars[:,:,max_ind])
                self.hyp_weight.append(weights[max_ind])
                self.miss_det.append( max_ind == 0 )
                self.miss_det_mean.append(means[:,0])
                self.miss_det_cov.append(covars[:,:,0])
                
                # Reduce mixture of states to one posterior estimate Gaussian.
                post_mean, post_covar = gm_reduce_single(means, covars, weights)
            
                # Add a Gaussian state approximation to the track.
                self.trk.append(GaussianStateUpdate(
                    post_mean, post_covar,
                    hypotheses,
                    hypotheses[0].measurement.timestamp))
    
                # Read the detections (index0 is timestep, index1 are the detections)
                det  = measurements
                im0s = self._yolo_detector.im0s        
                # Read frame
                pixels = copy(im0s) #copy(frame.pixels)
                # convert from CV2's BGR to RGB
                im_rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    
                # Plot output
                image = Image.fromarray(im_rgb)
                image = draw_detections(image, det)
                image = draw_tracks(image, [self.trk[-1]], idx, null_hyp=only_prediction)
                ax3.axes.xaxis.set_visible(False)
                ax3.axes.yaxis.set_visible(False)
                fig3.tight_layout()
                artist = ax3.imshow(image, animated=True)
                artists3.append([artist])
                
                # store it for plotting
                measurement_set = set()
                postulated_detection = 1 # missed detection won't count
                # how do you select specific item in a set?
                for meas in measurements:
                    if(postulated_detection == max_ind):
                        measurement_set.add(meas)
                    else:
                        measurement_set.add(Clutter(meas.state_vector, timestamp=meas.timestamp,
                                    measurement_model=meas.measurement_model))
                    postulated_detection += 1
                # for meas in measurements:
                #     measurement_set.add(meas)
                all_measurements.append(measurement_set)

                idx += 1
            
            if (num_frames) % 10:
                print("Frame: {}/{}".format(num_frames, num_frames))
            im_size = im0s.shape
            self.plotter.ax.set_xlim(0, im_size[1])
            self.plotter.ax.set_ylim(0, im_size[0])
            
            self.all_measurements = copy(all_measurements)
            # return 1
            # Plot true detections and clutter.
            self.plotter.plot_measurements(all_measurements, [0, 1]) # x,y
            # import pdb; pdb.set_trace()
            self.plotter.plot_tracks(self.trk, [0, 2], uncertainty=True) # x,y
            self.plotter.fig.tight_layout()
            self.plotter.fig
            # self.plotter.fig.show()
            # time.sleep(1.0)
            plt.pause(0.001)
            # plt.show()
            self.plotter.fig.savefig('tracker_output_PDA.png')
            print('saved the static plot tracker_output_PDA.png')
            ani3 = animation.ArtistAnimation(fig3, artists3, interval=33, \
                                      blit=True, repeat_delay=200)
            plt.pause(0.1)
            print('Saving to mpeg ...')
            progress_callback = lambda i,n: print(f'Saving frame {i} of {num_frames}') if i % 10 == 0 else None
            ani3.save('tracker_output_PDA.mp4', progress_callback=progress_callback)

if __name__ == '__main__':
    # fr = LoadVideoFromYouTube(30, 4*60, VIDEO_FILENAME='test1', \
                              # URL='https://www.youtube.com/watch?v=fkps18H3SXY')
    MOVT = None
    MOVT = MultiObjectVideoTracker(est_kind='PDA') #'multi')
    MOVT.run()
    print('done')
    # import pdb; pdb.set_trace()
    MOVT.plotter.fig
    hyp_mean = MOVT.hyp_mean
    hyp_cov  = MOVT.hyp_cov
    hyp_weight = MOVT.hyp_weight
    miss_det = MOVT.miss_det
    miss_det_mean = MOVT.miss_det_mean 
    miss_det_cov = MOVT.miss_det_cov
    all_measurements = MOVT.all_measurements
    trk = MOVT.trk