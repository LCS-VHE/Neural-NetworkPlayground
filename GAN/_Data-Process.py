import os
import time
import face_recognition
import numpy as np
from multiprocessing import Process
from PIL import Image

"""
Global configs, and directory
"""
CONCURRENT_PROCESS = 20

image_dir = os.path.join(os.curdir, "img_align_celeba")
images_dir = [os.path.join(image_dir, img_dir) for img_dir in os.listdir(image_dir)]
save_dir = os.path.join(os.curdir, "Cropped")


try:
    os.mkdir(save_dir)
except Exception as e:
    pass

print(f"Image_Dir : {len(images_dir)}, Process_Workload : {len(images_dir) / CONCURRENT_PROCESS}")
"""
Multiprocessing Cropping Images
"""


def save_image(image, filename):
    global save_dir
    image.resize((64,64)).save(os.path.join(save_dir, filename)) #Saving Image at path with filename


def crop_images(image_paths):
    for image in image_paths:
        try:
            img = Image.open(image)
            face_loc = face_recognition.face_locations(np.array(img))  # in format top right bottom left
            for loc in face_loc:
                img = img.crop((loc[3], loc[0], loc[1], loc[2])) # (left, top, right, bottom)
                save_image(img, image.split('/')[-1])
                break

        except Exception as e:
            print(f"Did not crop image {image}")


jump_count = int(len(images_dir)/CONCURRENT_PROCESS)
dirs =[images_dir[count * jump_count : (count+1) * jump_count] for count in range(CONCURRENT_PROCESS)]
processes = []
start = time.time()
for images_path in dirs:
    processes.append(
    Process(target=crop_images, args=(images_path, ))
    )

for process in processes:
    process.start()

for process in processes:
    process.join()

print(f"Program Done in {time.time() - start}")
