import os
import sys
from shutil import copy
from random import shuffle, choice


NUM_TRAIN = 7
NUM_VAL = 1
NUM_TEST = 2

chaser = [0] * NUM_TRAIN + [1] * NUM_VAL + [2] * NUM_TEST

#Directory that contains target files
source = 'source'

person = sys.argv[1]

#Destination directories
train = 'train'
validation = 'validation'
test = 'test'

#Create directories
os.mkdir(os.path.join(train,person))
os.mkdir(os.path.join(test,person))
os.mkdir(os.path.join(validation,person))

#Only works with file names
## t0 t1 t2 t3 etc.

#Split files into train/test
"""
labels = []

for i in range(NUM_FILES):
    if(i < NUM_TRAIN):
        labels.append(0)
    else:
        labels.append(1)

#Randomly shuffle file destinations
shuffle(labels)
"""
source = os.path.join(source,person)

for fileName in os.listdir(source):
    #Change the target file name
    
    #Create the path of the file
    path = os.path.join(source,fileName)

    #Init destination
    destination = None

    #Randomly set destination based on 40%/60% train-test split
    rand = choice(chaser)

    #Choose which directory to move file
    if rand == 0:
        destination = train
    elif rand == 2:
        destination = test
    elif rand == 1:
        destination = validation
    
    destination = os.path.join(destination,person)

    #Move file
    copy(path,destination)
