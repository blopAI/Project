# ParkGuard Pro
Project focused on enhancing parking experiences using artificial intelligence. 
We developed algorithms to recognize parking lines and spaces from drone images, made detection software for traffic sign recognition and used LiDAR to detect obstacles for safer parking.
We aimed to improve road safety using iPhone's LiDAR and a homemade drone, offering features like precise parking guidance, obstacle detection, and assisting drivers with special needs.

## Create a Virtual Environment:
```bash
$ python3 -m venv .venv
```
## Activate the Virtual Environment:
```bash
$ source .venv/bin/activate
```
## Setting up Redis:
To launch a Redis database container:
```bash
run sudo docker-compose up -d
```
```bash
$ sudo docker run -d --rm --name redis -p 6379:6379 redis:7.0
```
Check the container status:
```bash
$ sudo docker container list
```
Stop the container:
```bash
$ sudo docker container stop redis
```
For more information and features:
[Our documentation](https://github.com/blopAI/Project/wiki/Vzpostavitev-Redisa)
