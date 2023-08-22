import os

FRAMES = 50 # <-kejko frame-ov zelimo
STEP = 10   # <-Na koliko frame-ov ga bomo izrezali
VID_ROOT = 'data/videos/'
OUT_ROOT = 'data'

img_num = 24 # <-Nastavi na tejko ko hoces dat ime naslednji sliki
vid_file = 'vid_record_2023_5_6_6.mp4'
for i in range(img_num, FRAMES + img_num, STEP):
    bash_command = f'ffmpeg -i {VID_ROOT + vid_file} -vf "select=eq(n\,{i})" -vframes 1 img{img_num}.png'
    os.system(bash_command)
    img_num += 1
