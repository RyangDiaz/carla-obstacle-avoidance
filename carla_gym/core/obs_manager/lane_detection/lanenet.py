import numpy as np
import copy
import weakref
import carla
import torch
from queue import Queue, Empty
import albumentations as A
from albumentations.pytorch import ToTensorV2
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from carla_gym.core.obs_manager.lane_detection.utils.lane_detection.lane_detector import LaneDetector
from gymnasium import spaces

'''
    Gets binary segmentations of detected lanes using pretrained LaneNet model
'''

class ObsManager(ObsManagerBase):
    """
    Template config
    obs_configs = {
        "model_path": "/home/diaz0329/REU/carla-roach/carla_gym/core/obs_manager/lane_detection/lanenet_lane_detection_pytorch/log/carla_gym/core/obs_manager/lane_detection/lanenet_lane_detection_pytorch/log/loss=0.1223_miou=0.5764_epoch=73.pth",
        "camera_location": [-5.5, 0, 2.8],
        "camera_rotation": [0, -15, 0],
    }
    """

    def __init__(self, obs_configs):
        self._model_path = obs_configs['model_path']
        self._parent_actor = None
        self._world = None

        # Load in pretrained LaneNet model
        self._model = LaneDetector(model_path=self._model_path)
        print("Successfully loaded LaneNet model!")

        self.transform = A.Compose([
                A.Resize(256, 512),
                A.Normalize(),
                ToTensorV2()
            ])

        self._height = self._model.cg.image_height
        self._width = self._model.cg.image_width
        self._fov = self._model.cg.field_of_view_deg

        self._camera = None

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
            'lanes': spaces.Box(low=0, high=1, shape=(self._height,self._width), dtype=np.uint8),
            'frame': spaces.Discrete(2**32-1)
        })
    
    # Create camera and attach to ego vehicle
    def attach_ego_vehicle(self, parent_actor):
        # Create and load in camera
        self._image_queue = Queue()
        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        bp.set_attribute('fov', str(self._fov))

        self._camera = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._camera.listen(lambda image: self._parse_image(weak_self, image))
    
    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1

        try: 
            frame, data = self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('RGB sensor for LaneNet took too long!')
        
        with torch.no_grad():
            image = self.process_image(data)
            img = self._model(image)
        
        obs = {
            'frame': frame,
            'lanes': img
        }

        return obs
    
    def process_image(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = self.transform(image=image)['image']
        image = image.unsqueeze(0).to('cuda:0')
        return image

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

