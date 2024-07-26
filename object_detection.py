import sys
import argparse
import gc

from jetson_inference import detectNet
from jetson.utils import videoSource, videoOutput, Log
import jetson.utils

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a video stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video in and out
input = videoSource(opt.input, argv=["--input-timeout=5000"])
output= videoOutput(opt.output)
# output = videoOutput(opt.output,
#                     argv=sys.argv,
#                     options={
#                        'bitrate': 2500000,
#                        'codec': 'h264'
#                     })


# load the object detection network
net = detectNet(opt.network, sys.argv, opt.threshold)

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()
    if img is None:
        continue

    imgOutput = jetson.utils.cudaAllocMapped(width=img.width * 0.5, height=img.height * 0.5, format=img.format)
    # rescale the image (the dimensions are taken from the image capsules)
    jetson.utils.cudaResize(img, imgOutput)

    del img

    # detect objects in the image (with overlay)
    detections = net.Detect(imgOutput, overlay=opt.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)

    # render the image
    output.Render(imgOutput)

    del imgOutput
    gc.collect()

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # jetson.utils.cudaDeviceSynchronize()

    # check if the user quit
    if not input.IsStreaming():
        break

