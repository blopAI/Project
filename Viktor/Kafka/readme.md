# terminology
- Producer - producing messages; Consumer - consuming messages
- Producer and Consumer are decoupled
- Producer is producing, Consumer is consuming; If Consumer stops consuming, Producer can continue to produce
- After Consumer starts consuming again, it can continue on a point where it stopped (**offset**) - offset is stored in consumer
- The sequence of messages arriving at the Kafka Cluster is called **topic**; We can have as many topics as it is needed to describe various types of events
- A topic is subdivided into multiple storage units, known as **partitions** 
- A partition can be consumed by one or more consumers (can read at different offsets)
- Consumers can be **gourped**, ensuring a message is only read by a one consumer in the group - consumers can parallelize message processing

- More: https://medium.com/event-driven-utopia/understanding-kafka-topic-partitions-ae40f80552e8
        https://www.cloudkarafka.com/blog/part1-kafka-for-beginners-what-is-apache-kafka.html
        https://medium.com/nerd-for-tech/a-basic-introduction-to-kafka-a7d10a7776e6
        https://medium.com/@ahmedgulabkhan/3-simple-steps-to-set-up-kafka-locally-using-docker-b07f71f0e2c9

![Kafka](./kafka.png)

# Kafka installation
sudo apt update
sudo apt install default-jre
sudo apt install openjdk-8-jre-headless

wget https://dlcdn.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz

tar -xvzf kafka_2.13-3.4.0.tgz

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
export KAFKA_HOME=~/kafka_2.13-3.4.0/
source ~/.bashrc

# modify config file if in WSL
kafka/config/server.properties:
    uncomment and set IP in line: listeners=PLAINTEXT://172.31.146.134:9092
    umcomment line: listener.security.protocol.map=PLAINTEXT:PLAINTEXT,SSL:SSL,SASL_>

# start server
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties

# Better to do it with Docker

## Start Redis
sudo docker run -d --rm --name redis -p 6379:6379 redis:7.0

## Kafka
create file **docker-compose.yaml** with content:

version: '3'

services:
  zookeeper:
    image: wurstmeister/zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: localhost
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181

run sudo docker-compose up -d

run sudo docker exec kafka **command**

## General docker
sudo docker container list
sudo docker container stop **container**

# COMMANDS

## add topic
./kafka-topics.sh --bootstrap-server localhost:9092 --topic test_text --create --partitions 1 --replication-factor 1

## optional

### modify retention time
./kafka-configs.sh --alter --add-config retention.ms=2000 --bootstrap-server=172.31.146.134:9092 --topic test_text 

### modify num. of partitions
bin/kafka-topics.sh --zookeeper zk_host:port/chroot --alter --topic my_topic_name --partitions 40 

### modify max message size
./kafka-configs.sh --bootstrap-server 172.31.146.134:9092 --alter --entity-type topics --entity-name test_text --add-config max.message.bytes=10485880