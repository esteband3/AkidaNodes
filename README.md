# AkidaNodes

These are two versions of what the Raspberry Pi can run


The "hybrid" version uses OWL-ViT as its model and falls back on OpenCV's contour analysis

The "box_detector" uses the Akida API to build a model that takes advantage of the Akida's true processing potential

^This is overly difficult to get running, I've been trying for three days >:( (Ok scratch that, its working now YAYYYY)


The one we will be using for the competition is "camera_box_detector.py"


To run the program you will need to cd Documents then cd github. Then activate the venv by "source venv/bin/activate" then you can run the program with "python3 camera_box_detector.py" since the Akida requires Python 3.12 to run.


Only thing that is really left to accomplish on the program is to generate some kind of output and to make sure it boots upon startup.

