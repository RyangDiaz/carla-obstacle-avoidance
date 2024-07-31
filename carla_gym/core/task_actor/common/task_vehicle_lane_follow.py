import carla
import weakref
from .navigation.global_route_planner import GlobalRoutePlanner
from .navigation.route_manipulation import location_route_to_gps, downsample_route
import numpy as np
import logging
import copy

from .criteria import blocked, collision, outside_route_lane, route_deviation, run_stop_sign, timer
from .criteria import encounter_light, run_red_light

import carla_gym.utils.transforms as trans_utils
from carla_gym.core.obs_manager.object_finder.vehicle import ObsManager as OmVehicle
from carla_gym.core.obs_manager.object_finder.pedestrian import ObsManager as OmPedestrian
from carla_gym.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TaskVehicle(object):

    def __init__(self, vehicle, target_transforms, spawn_transforms, endless):
        """
        vehicle: carla.Vehicle
        target_transforms: list of carla.Transform
        """
        self.vehicle = vehicle
        world = self.vehicle.get_world()
        self._map = world.get_map()
        self._world = world

        self.criteria_blocked = blocked.Blocked()
        self.criteria_collision = collision.Collision(self.vehicle, world)
        self.criteria_light = run_red_light.RunRedLight(self._map)
        self.criteria_encounter_light = encounter_light.EncounterLight()
        self.criteria_stop = run_stop_sign.RunStopSign(world)
        self.criteria_outside_route_lane = outside_route_lane.OutsideRouteLane(self._map, self.vehicle.get_location())
        self.criteria_route_deviation = route_deviation.RouteDeviation()

        self.static_block_detector = blocked.Blocked(below_threshold_max_time=15.0)

        # navigation
        self._route_completed = 0.0
        self._route_length = 0.0

        self._target_transforms = target_transforms  # transforms

        self._planner = GlobalRoutePlanner(self._map, resolution=1.0)

        self._global_route = []
        self._global_plan_gps = []
        self._global_plan_world_coord = []

        self._trace_route_to_global_target()

        self._spawn_transforms = spawn_transforms

        self._endless = endless
        if len(self._target_transforms) == 0:
            while self._route_length < 15.0:
                self._add_random_target()

        self._last_route_location = self.vehicle.get_location()
        self.collision_px = False

        # For determining status of avoiding static obstacles
        self.avoid_obstacles = False
        self.return_to_road = False
        self.om_vehicle = OmVehicle({'max_detection_number': 10, 'distance_threshold': 20})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 10, 'distance_threshold': 20})
        self.om_vehicle.attach_ego_vehicle(self)
        self.om_pedestrian.attach_ego_vehicle(self)

        self.rtr_timer = timer.Timer()

    def _update_leaderboard_plan(self, route_trace):
        plan_gps = location_route_to_gps(route_trace)
        ds_ids = downsample_route(route_trace, 50)

        self._global_plan_gps += [plan_gps[x] for x in ds_ids]
        self._global_plan_world_coord += [(route_trace[x][0].transform.location, route_trace[x][1]) for x in ds_ids]

    def _add_random_target(self):
        if len(self._target_transforms) == 0:
            last_target_loc = self.vehicle.get_location()
            ev_wp = self._map.get_waypoint(last_target_loc)
            next_wp = ev_wp.next(6)[0]
            new_target_transform = next_wp.transform
        else:
            last_target_loc = self._target_transforms[-1].location
            last_road_id = self._map.get_waypoint(last_target_loc).road_id
            new_target_transform = np.random.choice([x[1] for x in self._spawn_transforms if x[0] != last_road_id])

        route_trace = self._planner.trace_route(last_target_loc, new_target_transform.location)
        self._global_route += route_trace
        self._target_transforms.append(new_target_transform)
        self._route_length += self._compute_route_length(route_trace)
        self._update_leaderboard_plan(route_trace)

    def _trace_route_to_global_target(self):
        current_location = self.vehicle.get_location()
        for tt in self._target_transforms:
            next_target_location = tt.location
            route_trace = self._planner.trace_route(current_location, next_target_location)
            self._global_route += route_trace
            self._route_length += self._compute_route_length(route_trace)
            current_location = next_target_location

        self._update_leaderboard_plan(self._global_route)

    @staticmethod
    def _compute_route_length(route):
        length_in_m = 0.0
        for i in range(len(route)-1):
            d = route[i][0].transform.location.distance(route[i+1][0].transform.location)
            length_in_m += d
        return length_in_m

    def _truncate_global_route_till_local_target(self, windows_size=5):
        ev_location = self.vehicle.get_location()
        closest_idx = 0

        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i+1][0].transform.location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i+1

        distance_traveled = self._compute_route_length(self._global_route[:closest_idx+1])
        self._route_completed += distance_traveled

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].transform.location)

        self._global_route = self._global_route[closest_idx:]
        return distance_traveled

    def _is_route_completed(self, percentage_threshold=0.99, distance_threshold=10.0):
        ev_loc = self.vehicle.get_location()

        percentage_route_completed = self._route_completed / self._route_length
        is_completed = percentage_route_completed > percentage_threshold
        is_within_dist = ev_loc.distance(self._target_transforms[-1].location) < distance_threshold

        return is_completed and is_within_dist
    
    # Determine how to set self.avoid_obstacles and self.return_to_road
    def _check_avoid_and_return(self, timestamp):
        # Check for potential static obstacles to avoid
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()

        # Check for blocking obstacles
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=20.0, up_angle_th=15)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=20.0)

        # Calculating overall deviation from route
        ev_transform = self.vehicle.get_transform()
        wp_transform = self.get_route_transform()

        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_distance = np.abs(np.dot(np_wp_unit_right, np_d_vec))
        angle_difference = np.deg2rad(np.abs(trans_utils.cast_angle(
            ev_transform.rotation.yaw - wp_transform.rotation.yaw)))

        # Check if lane change is available
        lane_change = False
        if len(self._global_route) > 8:
            current_location = self.vehicle.get_location()
            wp_ahead = self._global_route[8][0]
            lane_change = (wp_ahead.left_lane_marking.lane_change is not carla.LaneChange.NONE) or (wp_ahead.right_lane_marking.lane_change is not carla.LaneChange.NONE)
        
        # Avoid obstacles if hazards detected and no lane change available
        if not self.avoid_obstacles:
            if (hazard_vehicle_loc is not None or hazard_ped_loc is not None) and lane_change:
                self.avoid_obstacles = True
                self.return_to_road = False

        # When avoiding obstacles, check if safe to return to the route
        if not self.return_to_road and self.avoid_obstacles:
            # Return to route if no hazards detected
            self.return_to_road = hazard_vehicle_loc is None and hazard_ped_loc is None

            # Turn on timer if return to road activated
            if self.return_to_road:
                self.rtr_timer.turn_on(timestamp)

        if self.return_to_road:
            # Cannot be avoiding obstacles while returning to the road
            self.avoid_obstacles = False
        
            # Check timeout or back on route for return to road
            if self.rtr_timer.tick(timestamp) is not None or (lateral_distance < 0.2 and angle_difference < 0.03):
                self.return_to_road = False
                self.rtr_timer.turn_off()
    
    def _clear_surrounding_actors(self):
        ev_transform = self.vehicle.get_transform()
        ev_location = ev_transform.location
        def dist_to_ev(w): return w.get_location().distance(ev_location)

        surrounding_actors = []
        v_list = self._world.get_actors().filter("*vehicle*")
        p_list = self._world.get_actors().filter("*walker*")
        for actor_list in [v_list, p_list]:
            for ac in actor_list:
                has_different_id = self.vehicle.id != ac.id
                is_within_distance = dist_to_ev(ac) <= 20 # Distance threshold = 20
                if has_different_id and is_within_distance:
                    surrounding_actors.append(ac)
        
        sorted_surrounding_actors = sorted(surrounding_actors, key=dist_to_ev)
        if len(sorted_surrounding_actors) > 0:
            success = sorted_surrounding_actors[0].destroy()
            if success:
                print('[DEBUG] Destroyed blocking actor')
            else:
                print('[DEBUG] Tried to destroy actor but failed!')
        else:
            print('[DEBUG] Tried to destroy actor but none were found!')


    def tick(self, timestamp):
        distance_traveled = self._truncate_global_route_till_local_target()
        route_completed = self._is_route_completed()
        if self._endless and (len(self._global_route) < 10 or route_completed):
            self._add_random_target()
            route_completed = False
        
        self._check_avoid_and_return(timestamp)

        info_blocked = self.criteria_blocked.tick(self.vehicle, timestamp)
        info_collision = self.criteria_collision.tick(self.vehicle, timestamp)
        info_light = self.criteria_light.tick(self.vehicle, timestamp)
        info_encounter_light = self.criteria_encounter_light.tick(self.vehicle, timestamp)
        info_stop = self.criteria_stop.tick(self.vehicle, timestamp)
        info_outside_route_lane = self.criteria_outside_route_lane.tick(self.vehicle, timestamp, distance_traveled)
        info_route_deviation = self.criteria_route_deviation.tick(
            self.vehicle, timestamp, self._global_route[0][0], distance_traveled, self._route_length)
        
        static_block = self.static_block_detector.tick(self.vehicle, timestamp)

        if static_block: # and not (self.avoid_obstacles or self.return_to_road): # TODO TEMP
            self._clear_surrounding_actors()
            self.static_block_detector._time_last_valid_state = None
        
        # Don't count lane deviations if avoiding obstacles
        if self.avoid_obstacles or self.return_to_road:
            info_outside_route_lane = None

        info_route_completion = {
            'step': timestamp['step'],
            'simulation_time': timestamp['relative_simulation_time'],
            'route_completed_in_m': self._route_completed,
            'route_length_in_m': self._route_length,
            'is_route_completed': route_completed
        }

        self._info_criteria = {
            'route_completion': info_route_completion,
            'outside_route_lane': info_outside_route_lane,
            'route_deviation': info_route_deviation,
            'blocked': info_blocked,
            'collision': info_collision,
            'run_red_light': info_light,
            'encounter_light': info_encounter_light,
            'run_stop_sign': info_stop
        }

        # turn on light
        weather = self._world.get_weather()
        if weather.sun_altitude_angle < 0.0:
            vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        else:
            vehicle_lights = carla.VehicleLightState.NONE
        self.vehicle.set_light_state(carla.VehicleLightState(vehicle_lights))

        return self._info_criteria

    def clean(self):
        self.criteria_collision.clean()
        self.vehicle.destroy()

    @property
    def info_criteria(self):
        return self._info_criteria

    @property
    def dest_transform(self):
        return self._target_transforms[-1]

    @property
    def route_plan(self):
        return self._global_route

    @property
    def global_plan_gps(self):
        return self._global_plan_gps

    @property
    def global_plan_world_coord(self):
        return self._global_plan_world_coord
    
    @property
    def global_route(self):
        return self._global_route

    @property
    def route_length(self):
        return self._route_length

    @property
    def route_completed(self):
        return self._route_completed

    def get_route_transform(self):
        loc0 = self._last_route_location
        loc1 = self._global_route[0][0].transform.location

        if loc1.distance(loc0) < 0.1:
            yaw = self._global_route[0][0].transform.rotation.yaw
        else:
            f_vec = loc1 - loc0
            yaw = np.rad2deg(np.arctan2(f_vec.y, f_vec.x))
        rot = carla.Rotation(yaw=yaw)
        return carla.Transform(location=loc0, rotation=rot)
