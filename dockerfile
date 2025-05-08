FROM docker.hobot.cc/dlp/base:ubuntu22.04-gcc11.4-py3.10-cuda11.8

docker run --name zzy_test1 -d -p 10326:10326 --network=host --ipc=host -e gpus=all -v /home/users/zhiyu.zheng/datasets:/home/datasets/ --privileged docker.hobot.cc/dlp/base:ubuntu22.04-gcc11.4-py3.10-cuda11.8  tail -f /dev/null

docker run --name zzy_b2d_v2 -d -p 10326:10326 --network=host --ipc=host --runtime=nvidia --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v /home/users/zhiyu.zheng/datasets:/home/datasets/ -v /horizon-bucket:/horizon-bucket/ --privileged docker.hobot.cc/dlp/base:ubuntu22.04-gcc11.4-py3.10-cuda11.8  tail -f /dev/null


docker run --name zzy_test2 -d -p 10324:10324 --network=host --ipc=host --runtime=nvidia --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v /home/users/zhiyu.zheng/datasets:/home/datasets/ --privileged docker.hobot.cc/dlp/base:ubuntu22.04-gcc11.4-py3.10-cuda11.8  tail -f /dev/null

apt-get install ffmpeg libsm6 libxext6 libtiff5 libjpeg8 libjpeg62 libpng16-16

vim /etc/ssh/sshd_config
Port 10008
PermitRootLogin yes #允许root用户使用ssh登录
/etc/init.d/ssh restart
passwd


./contrib/download_prerequisites

mkdir build && cd build
../configure --prefix=/usr/local/gcc-9.4.0 --enable-languages=c,c++ --disable-multilib
sudo make -j32 && sudo make install