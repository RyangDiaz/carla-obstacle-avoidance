import numpy as np
import carla
import logging
from .zombie_vehicle import ZombieVehicle

FILTER_BLUEPRINTS = [
    'vehicle.carlamotors.carlacola',
    'vehicle.carlamotors.european_hgv',
    'vehicle.carlamotors.firetruck',
    'vehicle.tesla.cybertruck',
    'vehicle.ford.ambulance',
    'vehicle.mercedes.sprinter',
    'vehicle.volkswagen.t2',
    'vehicle.volkswagen.t2_2021',
    'vehicle.mitsubishi.fusorosa',
    'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja',
    'vehicle.vespa.zx125',
    'vehicle.yamaha.yzf',
    'vehicle.bh.crossbike',
    'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets'
]

class ZombieVehicleHandler(object):

    def __init__(self, client, tm_port=8000, spawn_distance_to_ev=4.0):
        self._logger = logging.getLogger(__name__)
        self.zombie_vehicles = {}
        self._client = client
        self._world = client.get_world()
        self._spawn_distance_to_ev = spawn_distance_to_ev
        self._tm_port = tm_port

    def reset(self, num_zombie_vehicles, ev_spawn_locations, seed=None):
        gen = np.random.default_rng(seed=seed)
        if type(num_zombie_vehicles) is list:
            n_spawn = np.random.randint(num_zombie_vehicles[0], num_zombie_vehicles[1])
        else:
            n_spawn = num_zombie_vehicles
        filtered_spawn_points = self._filter_spawn_points(ev_spawn_locations)
        # np.random.shuffle(filtered_spawn_points) # TODO: TEMP
        gen.shuffle(filtered_spawn_points)

        self._spawn_vehicles(filtered_spawn_points[0:n_spawn])

    def _filter_spawn_points(self, ev_spawn_locations):
        all_spawn_points = self._world.get_map().get_spawn_points()

        def proximity_to_ev(transform): return any([ev_loc.distance(transform.location) < self._spawn_distance_to_ev
                                                    for ev_loc in ev_spawn_locations])

        filtered_spawn_points = [transform for transform in all_spawn_points if not proximity_to_ev(transform)]

        return filtered_spawn_points

    def _spawn_vehicles(self, spawn_transforms):
        zombie_vehicle_ids = []
        # Filter vehicle blueprints (no bikes, no trucks/buses)
        all_blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        blueprints = []
        for bp in all_blueprints:
            if bp.id not in FILTER_BLUEPRINTS:
                blueprints.append(bp)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        spawned_transforms = []
        # Filter spawn transforms to make sure they aren't too close to each other
        print('[DEBUG] Attempting to spawn', len(spawn_transforms), 'vehicles!')
        for transform in spawn_transforms:
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'zombie_vehicle')

            # [TODO] TEMP: Toggle on/off autopilot for static/dynamic obstacles
            valid = True
            for st in spawned_transforms:
                dist = st.location.distance(transform.location)
                if dist < 30.0:
                    valid = False
                    # print('[INFO] Skipping vehicle spawn!')
                    break
            if valid:
                spawned_transforms.append(transform)
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, False, self._tm_port)))

        for response in self._client.apply_batch_sync(batch, do_tick=True):
            if not response.error:
                zombie_vehicle_ids.append(response.actor_id)

        for zv_id in zombie_vehicle_ids:
            self.zombie_vehicles[zv_id] = ZombieVehicle(zv_id, self._world)

        self._logger.debug(f'Spawned {len(zombie_vehicle_ids)} zombie vehicles. '
                           f'Should spawn {len(spawn_transforms)}')

    def tick(self):
        pass

    def clean(self):
        live_vehicle_list = [vehicle.id for vehicle in self._world.get_actors().filter("*vehicle*")]
        # batch1 = []
        # batch2 = []
        # SetAutopilot = carla.command.SetAutopilot
        # DestroyActor = carla.command.DestroyActor
        # batch1.append(SetAutopilot(zv_id, False))
        # batch1.append(DestroyActor(zv_id))
        # self._client.apply_batch_sync(batch1, do_tick=True)
        # self._client.apply_batch_sync(batch2, do_tick=True)
        for zv_id, zv in self.zombie_vehicles.items():
            if zv_id in live_vehicle_list:
                zv.clean()
        self.zombie_vehicles = {}
