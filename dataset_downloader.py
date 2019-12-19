import os
import json
from pprint import pprint
import pafy
'''
#### NOTE ####
We took this code from the following link to download the videos in the ActivityNet Dataset

https://github.com/ozgyal/ActivityNet-Video-Downloader/blob/master/activityNetDownloader.py

'''
# specify download directory
directory = 'Data/Original'
videoCounter = 0

# open json file
with open('Data/VideoLinks/activity_net.v1-3.min.json') as data_file:
    data = json.load(data_file)

# take only video informations from database object
videos = data['database']
print(videos)
# iterate through dictionary of videos
for key in videos:
    # take video
    video = videos[key]

    # find video subset
    subset = video['subset']

    # find video label
    annotations = video['annotations']
    label = ''
    if len(annotations) != 0:
        label = annotations[0]['label']
        label = '/' + label.replace(' ', '_')

    # create folder named as <label> if does not exist
    label_dir = directory + subset + label
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # take url of video
    url = video['url']

    # start to download
    try:
        video = pafy.new(url)
        best = video.getbest(preftype="flv")
        filename = best.download(filepath=label_dir + '/' + key)
        print ('Downloading... ' + str(videoCounter) + '\n')
        videoCounter += 1
    except Exception as inst:
        print ('Error!')
