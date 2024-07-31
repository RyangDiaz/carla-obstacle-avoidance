from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.carla_multi_agent_env import CarlaMultiAgentEnv
from carla_gym.utils import config_utils
import json


class SimpleRouteEnv(CarlaMultiAgentEnv):
    def __init__(self, carla_map, host, port, seed, no_rendering, obs_configs, reward_configs, terminal_configs,
                 weather_group, routes_group, render_mode=None):

        all_tasks = self.build_all_tasks(carla_map, weather_group, routes_group)
        super().__init__(carla_map, host, port, seed, no_rendering,
                         obs_configs, reward_configs, terminal_configs, all_tasks, render_mode)

    @staticmethod
    def build_all_tasks(carla_map, weather_group, routes_group):
        assert carla_map in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
        num_zombie_vehicles = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 250,
            'Town04': 150,
            'Town05': 50,
            'Town06': 120
        }
        num_zombie_walkers = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 100,
            'Town04': 80,
            'Town05': 40,
            'Town06': 80
        }

        # weather
        weathers = ['ClearNoon', 'ClearSunset']
        description_folder = CARLA_GYM_ROOT_DIR / 'envs/scenario_descriptions/AvoidObstacles' / carla_map

        actor_configs_dict = json.load(open(description_folder / 'actors.json'))
        route_descriptions_dict = config_utils.parse_routes_file(description_folder / 'routes.xml')

        all_tasks = []
        for weather in weathers:
            for route_id, route_description in route_descriptions_dict.items():
                task = {
                    'weather': weather,
                    'description_folder': description_folder,
                    'route_id': route_id,
                    'num_zombie_vehicles': num_zombie_vehicles[carla_map],
                    'num_zombie_walkers': num_zombie_walkers[carla_map],
                    'ego_vehicles': {
                        'routes': route_description['ego_vehicles'],
                        'actors': actor_configs_dict['ego_vehicles'],
                    },
                    'scenario_actors': {
                        'routes': route_description['scenario_actors'],
                        'actors': actor_configs_dict['scenario_actors']
                    } if 'scenario_actors' in actor_configs_dict else {}
                }
                all_tasks.append(task)

        return all_tasks
