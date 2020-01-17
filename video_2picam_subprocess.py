from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse

import socket
from imagezmq import ImageHub
import subprocess
import shlex 

def start_cam(cam = 'pi1', host = 'nassella', port = '8234'):
    try:
        remote_command = f'ssh -f {cam} ./startcam.sh {host}.local {port}'
        cmd = shlex.split(remote_command)
        subprocess.run(cmd)
    except (CalledProcessError) as exc:
        sys.exit(f'SSH connection to {cam} failed with: {exc}')


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    #old
    #videofile = args.video
    #cap = cv2.VideoCapture(videofile)
    #new
    picams = { 'pi1' : '8234', 'pi2' : '8235'} 
    thishost = socket.gethostname()
    
    print("bind ports for cameras")
    font = cv2.FONT_HERSHEY_SIMPLEX
    hubs = {}    
    for cam, port in picams.items():
        addr = socket.gethostbyname(f'{cam}.local')
        port_str = f'tcp://*:{port}'
        print(f'{thishost} waiting for {cam} at {port_str}')
        h = ImageHub(open_port=port_str)
        hubs[cam] = h
        cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
    
    frames = 0
    views = {}
    start = time.time()    


    for cam, port in picams.items():
        print(f'Starting {cam}')
        start_cam(cam = cam, host = thishost, port = port)

    #print('cams started')
go = True

try:
    while go:  # show streamed images until 'q'

        # get frames ~simultaneously
        for cam, h in hubs.items():
            #print(f'request image from {cam} {h}')
            msg, jpg_buffer = h.recv_jpg()
            frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
            h.send_reply(b'OK')
            views[cam] = (msg, frame)
            
        # process frames 
        for cam, (msg, frame) in views.items():
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
                
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            # if type(output) == int:
            #     frames += 1
            #     #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            #     cv2.putText(orig_im,
            #                     msg,(0,130), font, 1,
            #                     (200,255,155), 2, cv2.LINE_AA)
            #     cv2.imshow(cam, orig_im)
            #     key = cv2.waitKey(1)
            #     if key & 0xFF == ord('q'):
            #         break
            #     continue
            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            cv2.imshow(cam, orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                go = False
                break
            frames += 1
            now = time.time()
            print(str(frames) + " FPS of the video is {:5.2f}".format( frames / (now - start)))
            # reset fps
            if frames > 20: 
                frames = 0
                start = now     
    
    # TODO capture.release() all cams
finally:
    cv2.destroyAllWindows()
    exit(0)
    #

    

    
    

