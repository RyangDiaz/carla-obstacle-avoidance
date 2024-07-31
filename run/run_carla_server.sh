docker run --privileged --gpus all --net=host -p "3000-3002:3000-3002"  -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-port=3000 -world-port=3000
docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen
sudo killall -9 -r CarlaUE4-Linux