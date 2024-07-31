import numpy as np
import copy
import weakref
import carla
from queue import Queue, Empty
from gymnasium import spaces

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    Template configs:
    obs_configs = {
        "module": "camera.depth",
        "location": [-5.5, 0, 2.8],
        "rotation": [0, -15, 0],
        "frame_stack": 1,
        "width": 1920,
        "height": 1080
    }
    frame_stack: [Image(t-2), Image(t-1), Image(t)]
    """

    def __init__(self, obs_configs):

        self._sensor_type = 'camera.depth'

        self._height = obs_configs['height']
        self._width = obs_configs['width']
        self._fov = obs_configs['fov']
        self._channels = 1

        location = carla.Location(
            x=float(obs_configs['location'][0]),
            y=float(obs_configs['location'][1]),
            z=float(obs_configs['location'][2]))
        rotation = carla.Rotation(
            roll=float(obs_configs['rotation'][0]),
            pitch=float(obs_configs['rotation'][1]),
            yaw=float(obs_configs['rotation'][2]))

        self._camera_transform = carla.Transform(location, rotation)

        self._sensor = None
        self._queue_timeout = 10.0
        self._image_queue = None

        super(ObsManager, self).__init__()

    def _define_obs_space(self):

        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Box(
                low=0, high=1, shape=(self._height, self._width, self._channels), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        init_obs = np.zeros([self._height, self._width, self._channels], dtype=np.float32)
        self._image_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor."+self._sensor_type)
        bp.set_attribute('image_size_x', str(self._width))
        bp.set_attribute('image_size_y', str(self._height))
        bp.set_attribute('fov', str(self._fov))

        self._sensor = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda image: self._parse_depth(weak_self, image))

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._image_queue.qsize() <= 1

        try: 
            frame, data = self._image_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('Depth sensor took too long!')

        obs = {'frame': frame,
               'data': data}

        return obs

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        self._sensor = None
        self._world = None

        self._image_queue = None

    @staticmethod
    def _parse_depth(weak_self, carla_image):
        self = weak_self()

        carla_img.convert(carla.ColorConverter.Depth)
        np_img = np.array(carla_img.raw_data).reshape((carla_img.height,carla_img.width,4))[:,:,0] * 1000 / 255
        np_img = copy.deepcopy(np_img)
        np_img = np_img.astype(np.float32)

        self._image_queue.put((carla_image.frame, np_img))
