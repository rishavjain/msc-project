from mkdirs import mkdirs

import os
import time
import gzip
import math

PLATFORM = 'iceberg'

if PLATFORM == 1:
    OUT_PATH = 'e:/tmp'
    GZ_INPUT = '../tmp/ukwac_subset_1M.txt.gz'
elif PLATFORM == 2:
    OUT_PATH = '../tmp'
    GZ_INPUT = '../tmp/ukwac_subset_1M.txt.gz'
elif PLATFORM == 'iceberg':
    OUT_PATH = '/data/cop15rj/ukwac100'
    GZ_INPUT = '/data/cop15rj/downloads/ukwac_subset_100M.txt.gz'


OUT_FILENAME = 'ukwac'
OUT_NUMLINES = 500
CREATE_INSTANCE_FOLDER = False
FILELIST = 'filelist.txt'

GZ_INPUT = os.path.abspath(GZ_INPUT)
OUT_PATH = os.path.abspath(OUT_PATH)

if CREATE_INSTANCE_FOLDER:
    OUT_PATH = os.path.join(OUT_PATH, time.strftime('%m-%d_%H;%M;%S'))

print('OUT_PATH', ':', OUT_PATH)

mkdirs(OUT_PATH)

numFiles = 1  # index for generated input file

outputFileNames = ()

if OUT_NUMLINES:
    outputFile = open(os.path.join(OUT_PATH, OUT_FILENAME + str(numFiles)), 'w', encoding='iso-8859-15')
    fileList = open(os.path.join(OUT_PATH, FILELIST), 'w')
    print('fileList', ':', fileList.name)
else:
    outputFile = open(os.path.join(OUT_PATH, OUT_FILENAME), 'w', encoding='iso-8859-15')

print('outputFile', ':', outputFile.name)

numLines = 0
inputFile = gzip.open(GZ_INPUT, mode='rt', encoding='ISO-8859-15')
print('inputFile', ':', inputFile.name)

for line in inputFile:
    if not line.startswith('CURRENT URL'):

        outputFile.write(line)
        numLines += 1

        if OUT_NUMLINES and numLines % OUT_NUMLINES == 0:
            outputFile.close()
            outputFileNames += (outputFile.name,)

            fileList.write(os.path.abspath(outputFile.name) + '\n')
            fileList.flush()

            numFiles += 1

            outputFile = open(os.path.join(OUT_PATH, OUT_FILENAME + str(numFiles)), 'w', encoding='iso-8859-15')
            print('outputFile', ':', outputFile.name)

outputFile.close()


if OUT_NUMLINES and math.ceil(numLines / OUT_NUMLINES) < numFiles and numLines % OUT_NUMLINES == 0:
    os.remove(outputFile.name)
    numFiles -= 1
else:
    outputFileNames += (os.path.abspath(outputFile.name),)
    fileList.write(os.path.abspath(outputFile.name))

print('numFiles', ':', numFiles)
print('numLines', ':', numLines)

print('outputFileNames', ':')
for file in outputFileNames:
    print(file)
