import pygame

def check_intersection(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left1 = x1 - w1 / 2
    right1 = x1 + w1 / 2
    top1 = y1 - h1 / 2
    bottom1 = y1 + h1 / 2

    left2 = x2 - w2 / 2
    right2 = x2 + w2 / 2
    top2 = y2 - h2 / 2
    bottom2 = y2 + h2 / 2

    if left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2:
        return True
    else:
        return False


pygame.init()
screen_width = 1344
screen_height = 1008
screen = pygame.display.set_mode((screen_width, screen_height))
background_image = pygame.image.load("carparking1.6.2023/IMG_9814.JPG").convert()
background_image = pygame.transform.rotate(background_image, -90)
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))
color = (230, 1, 54)

def draw_dot(rectangles):
    for rect in rectangles:
        x, y, width, height = rect
        top_left = (int(x - width / 2), int(y - height / 2))
        top_right = (int(x + width / 2), int(y - height / 2))
        bottom_left = (int(x - width / 2), int(y + height / 2))
        bottom_right = (int(x + width / 2), int(y + height / 2))

        pygame.draw.circle(screen, color, top_left, 10)
        pygame.draw.circle(screen, color, top_right, 10)
        pygame.draw.circle(screen, color, bottom_left, 10)
        pygame.draw.circle(screen, color, bottom_right, 10)


def read_annotations_from_file(filename):
    annotations = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split()
            label = int(values[0])
            x = float(values[1])
            y = float(values[2])
            width = float(values[3])
            height = float(values[4])
            annotations.append([label, x, y, width, height])
    return annotations


annotations = read_annotations_from_file("carparkingDATA2.6.2023/obj_train_data/IMG_9814.txt")

bounding_boxes = []
actual_coordinates = []

for annotation in annotations:
    label = annotation[0]
    x = int(annotation[1] * screen_width)
    y = int(annotation[2] * screen_height)
    width = int(annotation[3] * screen_width)
    height = int(annotation[4] * screen_height)
    bounding_boxes.append((x, y, width, height))
    actual_coordinates.append((x, y))  # Append actual x, y coordinates

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(background_image, (0, 0))

    for box in bounding_boxes:
        x, y, width, height = box
        left = int(x - width / 2)
        top = int(y - height / 2)
        pygame.draw.rect(screen, (0, 0, 0), (left, top, width, height), 2)

    #draw_dot(bounding_boxes)

    last_box = bounding_boxes[-1]
    intersects = False

    for box in bounding_boxes[:-1]:
        if check_intersection(last_box, box):
            intersects = True
            break

    if intersects:
        pygame.draw.rect(screen, (255, 0, 0), (left, top, width, height), 2)
        print("Last bounding box intersects with other boxes.")
    else:
        pygame.draw.rect(screen, (0, 255, 0), (left, top, width, height), 2)
        print("Last bounding box does not intersect with other boxes.")

    pygame.display.update()

screenshot = pygame.display.get_surface()
pygame.image.save(screenshot, "screenshot56.png")

pygame.quit()
