# BLOB AI - Self Driving curb detection car project

## Structure

### Capturing Depth Images with iPhone LiDAR
This project is a curb detection self driving car system built as part of our **CS university diploma**. We scanned curbs along roads using the **iPhone 12 Pro Max LiDAR sensor** and wrote a custom Python script to convert the captured **3D point clouds** into **2D top down height maps**.

Example of height map:

The raw LiDAR point clouds contained over 100,000 vertices some even reached 300,000–500,000. To handle this, we implemented a custom decimation algorithm using Delaunay triangulation before constructing the height map. The results were surprisingly good and we preserved most of detail, Sadly we  discovered that this approach was slower than simply coloring points by height and projecting them directly into a 2D image :(.

Example of Delunay Triangulation:
<img width="600" alt="image" src="https://github.com/user-attachments/assets/4d63b9bf-cf7f-4a35-b304-60dc1143b4f6" />

### Curbs Detection

Then we used these **Height Maps** to train a **YOLOv5** model for curb detection.

<img width="400" alt="Screenshot 2026-01-27 at 11 12 43" src="https://github.com/user-attachments/assets/b80ae6cf-888e-4ec5-89a9-cce33c521e36" />

<img width="400" alt="Screenshot 2026-01-27 at 11 12 26" src="https://github.com/user-attachments/assets/4f17da2e-8eea-4ba2-9bd6-a18b54884b3c" />

<img width="600" alt="image" src="https://github.com/user-attachments/assets/db1b0dc5-d989-4021-a791-bc4aa170985a" />

### Building the Car with Arduino 
We created simple self driving car with **Arduino** and added **Ultrasonic** sensors alongside **wifi module** to send Coordinates to computer via **Kafka**


### 3D Simulation
We created a 3D environment using **JavaScript** and **Three.js** that visualized the real car’s position based on data from the ultrasonic sensor. When the car moved in real space, the simulation updated accordingly. We decided to use a little trick and didn’t move the car model directly inside the simulation, we instead moved the obstacle (cone & environment) soo it looked like car is moving. This part was used for nice visual representation on screen :D.

<img width="350" alt="Screenshot 2026-01-27 at 11 54 44" src="https://github.com/user-attachments/assets/a1131ebd-7fd9-4dae-9273-823aa05658bb" />

<img width="350" alt="Screenshot 2026-01-27 at 11 54 29" src="https://github.com/user-attachments/assets/f184f41b-b37c-4160-8611-8aaf0b816a03" />

<img width="350" alt="Screenshot 2026-01-27 at 11 54 07" src="https://github.com/user-attachments/assets/0f90203b-4865-46c0-9560-696aa5e79375" />

### Car Stopping Logic
To make **YOLO** handle the stopping logic, we added a **second camera** inside the same Three.js  scene. This camera wasn’t visible in the main view. It looked at a tiny setup with a black plane and a white cube that represented the curb.

We moved the white cube based on the real ultrasonic distance (same trick as with the cone).
The second camera sent its image to a Node.js  server through a **GET request**, where **YOLO ran detection**. If YOLO spotted the white cube, the server sent a message through **Kafka** and the car stopped. 

<img width="400" alt="Screenshot 2026-01-27 at 12 02 24" src="https://github.com/user-attachments/assets/1e711bc5-a52d-4d95-8592-c148b83760e1" />

<img width="400" alt="Screenshot 2026-01-27 at 12 02 59" src="https://github.com/user-attachments/assets/1a6d16a9-bc00-425e-9838-3a4bf23a6f04" />


