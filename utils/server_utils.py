import subprocess
import os
import time
from omegaconf import OmegaConf

import logging
log = logging.getLogger(__name__)


def kill_carla():
    # NOTE: DO NOT PUSH THIS, VERY INSECURE!
    sudo_password = '' # TODO: Put sudo password here!
    kill_process = subprocess.Popen(f'echo {sudo_password} | sudo -S killall -9 -r CarlaUE4-Linux', shell=True)
    kill_process.wait()
    time.sleep(1)
    log.info("Kill Carla Servers!")


class CarlaServerManager():
    def __init__(self, carla_sh_str, port=2000, configs=None, t_sleep=5):
        self._carla_sh_str = carla_sh_str
        self._t_sleep = t_sleep
        self.env_configs = []

        if configs is None:
            cfg = {
                'gpu': 0,
                'port': port,
            }
            self.env_configs.append(cfg)
        else:
            for cfg in configs:
                for gpu in cfg['gpu']:
                    single_env_cfg = OmegaConf.to_container(cfg)
                    single_env_cfg['gpu'] = gpu
                    single_env_cfg['port'] = port
                    self.env_configs.append(single_env_cfg)
                    port += 1000

    def start(self):
        kill_carla()
        for cfg in self.env_configs:
            # cmd = f'CUDA_VISIBLE_DEVICES={cfg["gpu"]} bash {self._carla_sh_str} ' \
            #     f'-fps=10 -quality-level=Epic -carla-rpc-port={cfg["port"]}'
            #     f'-fps=10 -carla-server -opengl -carla-rpc-port={cfg["port"]}'
            # if cfg["port"] < 3000:
            #     cmd = f'CUDA_VISIBLE_DEVICES={cfg["gpu"]} docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen'
            # else:
            cmd = f'docker run --net=host --gpus \'"device={cfg["gpu"]}"\' -p "{cfg["port"]}-{cfg["port"]+2}:{cfg["port"]}-{cfg["port"]+2}"' \
                f' -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-port={cfg["port"]} -world-port={cfg["port"]}'
            log.info(cmd)
            server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self._t_sleep)

    def stop(self):
        kill_carla()
        time.sleep(self._t_sleep)
        log.info(f"Kill Carla Servers!")
