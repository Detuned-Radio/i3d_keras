import cv2
import os
import shutil
import sys
import time
import numpy as np
import argparse

SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 224
FRAME_RATE = 25


# sample frames at 25 frames per second
def sample_video(video_path, path_output):
    # for filename in os.listdir(video_path):
    if video_path.endswith((".mp4", ".avi")):
        # filename = video_path + filename
        os.system(f"ffmpeg -r {FRAME_RATE} -i {video_path} -q:v 2 {path_output}/frame_%05d.jpg")
    else:
        raise ValueError("Video path is not the name of video file (.mp4 or .avi)")


# the videos are resized preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation
def resize_dims(img):
    original_width = int(img.shape[1])
    original_height = int(img.shape[0])

    aspect_ratio = original_width / original_height

    if original_height < original_width:
        new_height = SMALLEST_DIM
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = SMALLEST_DIM
        new_height = int(original_width / aspect_ratio)

    dim = (new_width, new_height)
    return dim

def resize(img):
    # resize image
    resized = cv2.resize(img, resize_dims(img), interpolation=cv2.INTER_LINEAR)
    return resized


def crop_center(img, new_size):
    y, x, c = img.shape
    (cropx, cropy) = new_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def rescale_pixel_values(img):
    # print('Data Type: %s' % img.dtype)
    # print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    img = img.astype('float32')
    # normalize to the range 0:1
    # img /= 255.0
    # normalize to the range -1:1
    img = (img / 255.0) * 2 - 1
    # confirm the normalization
    # print('Min: %.3f, Max: %.3f' % (img.min(), img.max()))
    return img


# The provided .npy file thus has shape (1, num_frames, 224, 224, 3) for RGB, corresponding to a batch size of 1
def run_rgb(sorted_list_frames, train):
    if not train:
        output_dims = (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
    else:
        output_dims = resize_dims(cv2.imread(sorted_list_frames[0], cv2.IMREAD_UNCHANGED))
    output_h = output_dims[0]
    output_w = output_dims[1]
    result = np.zeros((1, output_h, output_w, 3))
    for full_file_path in sorted_list_frames:
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
        img = pre_process_rgb(img, train)
        new_img = np.reshape(img, (1, output_h, output_w, 3))
        result = np.append(result, new_img, axis=0)

    result = result[1:, :, :, :]
    result = np.expand_dims(result, axis=0)
    return result


def pre_process_rgb(img, train):
    resized = resize(img)
    if not train:
        img_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    else:
        img_cropped = resized
    img = rescale_pixel_values(img_cropped)
    return img


def read_frames(video_path):
    list_frames = []
    for file in os.listdir(video_path):
        if file.endswith(".jpg"):
            full_file_path = video_path + file
            list_frames.append(full_file_path)
    sorted_list_frames = sorted(list_frames)
    return sorted_list_frames


def run_flow(sorted_list_frames, train, debug):
    sorted_list_img = []
    for frame in sorted_list_frames:
        img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sorted_list_img.append(img_gray)

    if not train:
        output_dims = (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
    else:
        output_dims = resize_dims(cv2.imread(sorted_list_img[0], cv2.IMREAD_UNCHANGED))
    output_h = output_dims[0]
    output_w = output_dims[1]
    result = np.zeros((1, output_h, output_w, 2))

    prev = sorted_list_img[0]
    frame_no = 1
    for curr in sorted_list_img[1:]:
        start = time.time()
        flow = compute_optical_flow(prev, curr)
        end = time.time()
        if debug:
            print(f'#{frame_no} Flow computation took: {end - start}')
        flow = pre_process_flow(flow, train)
        prev = curr
        result = np.append(result, flow, axis=0)
        frame_no += 1

    result = result[1:, :, :, :]
    result = np.expand_dims(result, axis=0)
    return result


def pre_process_flow(flow_frame, train):
    resized = resize(flow_frame)
    if not train:
        img_cropped = crop_center(resized, (IMAGE_CROP_SIZE, IMAGE_CROP_SIZE))
    else:
        img_cropped = resized
    new_img = np.reshape(img_cropped, (1, img_cropped.shape[0], img_cropped.shape[1], 2))
    return new_img


#  Pixel values are truncated to the range [-20, 20], then rescaled between -1 and 1
def compute_optical_flow(prev, curr):
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    flow_frame = optical_flow.calc(prev, curr, None)
    flow_frame = np.clip(flow_frame, -20, 20)
    flow_frame = flow_frame / 20.0
    return flow_frame

# ---> MAIN FUNCTION
def preprocess_video(video_path, save_path, train=False, debug=False):
    frame_output_path = '/tmp/frames/'
    if not os.path.exists(frame_output_path):
        os.makedirs(frame_output_path)

    # sample all video from video_path at specified frame rate (FRAME_RATE param)
    start = time.time()
    sample_video(video_path, frame_output_path)
    end = time.time()
    samp_dur = end - start
    if debug:
        print(f'Resampled video to 25fps. Time Taken: {samp_dur} sec')

    # make sure the frames are processed in order
    sorted_list_frames = read_frames(frame_output_path)
    if len(sorted_list_frames) == 0:
        print('ERROR: No frames read.')
        os.rmdir(frame_output_path)
        sys.exit()
    if debug:
        print(f'Read {len(sorted_list_frames)} frames.')

    video_name = video_path.split("/")[-1][:-4]
    class_name = video_path.split("/")[-2]

    if not os.path.exists(save_path + class_name + "/" ):
        os.makedirs(save_path + class_name + "/" )

    start = time.time()
    rgb = run_rgb(sorted_list_frames, train)
    npy_rgb_output = save_path + class_name + "/" + video_name + '_rgb.npy'
    np.save(npy_rgb_output, rgb)
    end = time.time()
    rgb_dur = end - start
    if debug:
        print(f'RGB output saved. Time taken: {rgb_dur} sec')

    start = time.time()
    flow = run_flow(sorted_list_frames, train, debug)
    npy_flow_output = save_path + class_name + "/" + video_name + '_flow.npy'
    np.save(npy_flow_output, flow)
    end = time.time()
    opt_dur = end - start
    if debug:
        print(f'Optical Flow output saved. Time taken: {opt_dur} sec')

    shutil.rmtree(frame_output_path)

