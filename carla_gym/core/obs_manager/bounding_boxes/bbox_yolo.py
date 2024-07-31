import numpy as np
import copy
import weakref
import carla
import torch
from queue import Queue, Empty
from ultralytics import YOLO
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from gymnasium import spaces

'''
    Gets detected bounding boxes using pre-trained YOLO model on RGB input
'''

class ObsManager(ObsManagerBase):
    """
    Template config
    obs_configs = {
        "model_path": "/home/diaz0329/REU/carla-roach/models/best.pt",
        "imgsz": 640,
        "conf": 0.5,
        "max_detection_number": 10,
        "camera_location": [-5.5, 0, 2.8],
        "camera_rotation": [0, -15, 0],
        "camera_width": 1920,
        "camera_height": 1080,
        "camera_fov": 10
    }
    """

    def __init__(self, obs_configs):
        self._model_path = obs_configs['model_path']
        self._imgsz = obs_configs['imgsz']
        self._conf = obs_configs['conf']
        self._max_detection_number = obs_configs['max_detection_number']
        self._parent_actor = None
        self._world = None

        self._height = obs_configs['camera_height']
        self._width = obs_configs['camera_width']
        self._fov = obs_configs['camera_fov']
        self._channels = 4

        self._camera = None # TODO: Create 
        self._model = None # TODO: Create

        location = carla.Location(
            x=float(obs_configs['camera_location'][0]),
            y=float(obs_configs['camera_location'][1]),
            z=float(obs_configs['camera_location'][2]))
        rotation = carla.Rotation(
            roll=float(obs_configs['camera_rotation'][0]),
            pitch=float(obs_configs['camera_rotation'][1]),
            yaw=float(obs_configs['camera_rotation'][2]))
        
        self._camera_transform = carla.Transform(location, rotation)

        self._sensor = None
        self._queue_timeout = 10.0
        self._image_queue = None

        super(ObsManager, self).__init__()
    
    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            # xywh format (normalized by image size)
            'bboxes': spaces.Box(low=0, high=1, shape=(self._max_detection_number,4), dtype=np.float32),
            'classes': spaces.MultiDiscrete(np.ones(self._max_detection_number) * 6.0, dtype=np.uint8),
            'frame': spaces.Discrete(2**32-1)
        })
    
    # Create both camera and model
    def attach_ego_vehicle(self, parent_actor):
        # Create and load in camera
        self._image_queue = Queue()
        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        bp.set_attribute('fov', str(self._fov))
        # set in leaderboard
        bp.set_attribute('lens_circle_multiplier', str(3.0))
        bp.set_attribute('lens_circle_falloff', str(3.0))
        bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        bp.set_attribute('chromatic_aberration_offset', str(0))

        self._camera = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._camera.listen(lambda image: self._parse_image(weak_self, image))

        # Load in pretrained YOLO model
        self._model = YOLO(self._model_path)
        print("Successfully loaded YOLO model!")
    
    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1

        try: 
            frame, data = self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('RGB sensor for bboxes took too long!')

        obs = {'frame': frame,
               'data': data}
        
        with torch.no_grad():
            results = self._model.predict(
                data, 
                conf=self._conf,
                imgsz=self._imgsz,
                max_det=self._max_detection_number,
                device=1
            )
        
        bboxes = results[0].boxes.xywhn.detach().cpu().numpy().astype(np.float32)
        classes = results[0].boxes.cls.detach().cpu().numpy().astype(np.uint8)+1 # Add 1 to classes so that 0 represents null class

        if bboxes.shape[0] == 0:
            bboxes = np.zeros(shape=(1,4)).astype(np.float32)
            classes = np.zeros(shape=(1,)).astype(np.uint8)

        # Fill in empty spots in obs with null detections
        for i in range(self._max_detection_number-bboxes.shape[0]):
            bboxes = np.append(bboxes, np.zeros(shape=(1,4)), axis=0)
            classes = np.append(classes, 0)
        
        obs = {
            'frame': frame,
            'bboxes': bboxes,
            'classes': classes
        }

        return obs

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        
        self._sensor = None
        self._world = None

        if self._model:
            del self._model
        
        self._model = None
        
    @staticmethod
    def _parse_image(weak_self, carla_image):
        self = weak_self()

        np_img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))

        np_img = copy.deepcopy(np_img)

        np_img = np.reshape(np_img, (carla_image.height, carla_image.width, 4))
        np_img = np_img[:, :, :3]

        self._image_queue.put((carla_image.frame, np_img))

