#!/usr/bin/env python3

# To make this work execute the following steps:
# sudo apt install ffmpeg acoustid-fingerprinter
# pip3 install pyacoustid
# pip3 install fuzzywuzzy[speedup]
# pip3 install tqdm

from multiprocessing import Process, Pool, Queue
from functools import partial
import io
import sys
import glob 
import acoustid
import chromaprint
from fuzzywuzzy import fuzz
from operator import itemgetter 
import csv
from tqdm import tqdm
import os
import numpy as np
import pickle

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
 
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return(matrix[size_x - 1, size_y - 1].tolist())


def progressBarMessenger(barSize, descripton):

    currentBarTicks = [] 
    progressBar = []
    for i in range(len(barSize)):
        currentBarTicks.append(0)
        progressBar.append(tqdm(total=barSize[i], file=sys.stdout))
        progressBar[i].set_description(descripton[i])
 
    for i in range(sum(barSize)):

        ind, message = messageQueue.get()
        if message:
            tqdm.write(message)
        progressBar[ind].update(1)
        currentBarTicks[ind] = currentBarTicks[ind] + 1

        for j in range(len(barSize)):
            if currentBarTicks[j] == barSize[j]:
                progressBar[j].close()

    for bar in progressBar:
        bar.close()

def getFingerprint():

    file = jobQueue.get()

    messageQueue.put([0, file])
    
    
    # Check if fingerprint already stored in temporary progress directory
    if os.path.exists(PROGRESS_PATH + file + '.p'):
        file_p = open(PROGRESS_PATH + file + '.p', 'rb')
        fingerprint = pickle.load(file_p)
        file_p.close()
        resultsQueue.put([file, fingerprint])
        
    else:

        try:
            duration, fp_encoded = acoustid.fingerprint_file(file)
            fingerprint, version = chromaprint.decode_fingerprint(fp_encoded)
            file_p = open(PROGRESS_PATH + file + '.p', 'wb')
            pickle.dump(fingerprint, file_p)
            file_p.close()
            resultsQueue.put([file, fingerprint])
        except Exception as e:
        
            if os.path.exists(LOG_PATH):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
        
            # Open txt file for logging
            log = open(LOG_PATH,append_write)

            log.writelines('Error: "' + str(e) + '" while processing file: ' + file + "\n")
            messageQueue.put([0, 'Error: "' + str(e) + '" while processing file: ' + file + "\n"])

            # Close log file
            log.close()

            resultsQueue.put([])
        finally:
            pass

def compareFingerprints():

    candidate1, candidate2 = jobQueue.get()
    
    messageQueue.put([1, candidate1[0] + ' vs ' + candidate2[0]])
    
    # Check if result for current candidates already processed
    if os.path.exists(PROGRESS_PATH + (candidate1[0] + '_vs_' + candidate2[0]).replace('.', '_').replace('/', '_') + '.p'):
        file_p = open(PROGRESS_PATH + (candidate1[0] + '_vs_' + candidate2[0]).replace('.', '_').replace('/', '_') + '.p', 'rb')
        [fingerprintSimilarity, levenshteinSimilariry] = pickle.load(file_p)
        file_p.close()
    
    else:
        fingerprintSimilarity = fuzz.ratio(candidate1[1], candidate2[1])
        levenshteinSimilariry = levenshtein(os.path.splitext(candidate1[0])[0], os.path.splitext(candidate2[0])[0])
        file_p = open(PROGRESS_PATH + (candidate1[0] + '_vs_' + candidate2[0]).replace('.', '_').replace('/', '_') + '.p', 'wb')
        pickle.dump([fingerprintSimilarity, levenshteinSimilariry], file_p)
        file_p.close()
		
    resultsQueue.put([candidate1[0], candidate2[0], fingerprintSimilarity, levenshteinSimilariry])

def newJobGenerator(rawData):
    
    comparisionPairs =[]        
    for candidate1_index in range(0, len(rawData)):
        for candidate2_index in range(candidate1_index+1, len(rawData)):
            messageQueue.put([0, ''])
            comparisionPairs.append([rawData[candidate1_index], rawData[candidate2_index]])
            jobQueue.put([rawData[candidate1_index], rawData[candidate2_index]])



if __name__ == "__main__":
    
    # Settings
    CPU_COUNT = os.cpu_count()
    MUSIC_PATH = './*'
    LOG_PATH = './log.txt'
    RESULTS_PATH = './results.csv'
    FILE_EXTENSIONS = ['mp3', 'aac', 'm4a', 'ogg', 'wav', 'flac', 'vorbis', 'opus']
    PROGRESS_PATH = './progress/'
    
    # Make directory to store temporary progress data
    if not os.path.exists(PROGRESS_PATH):
        os.mkdir(PROGRESS_PATH)
     
    # Fingerprint and other data of each processed files are saved in this list
    rawData = []

    # Fingerprint comparision results are stored here
    comparisionData = []
    
    # List of all audio files in specified dir
    fileList = []
    for EXTENSION in FILE_EXTENSIONS:
        fileList = fileList + glob.glob(MUSIC_PATH + '.' + EXTENSION)

    # Messages are passed to messageQueue
    messageQueue = Queue()

    # New jobs are passed to jobQueue
    jobQueue = Queue()

    # Queue for results
    resultsQueue = Queue()

    # Start new process to show progress over progress bar
    progressBarProcess = Process(target=progressBarMessenger, args=([len(fileList)], ['Fingerprinting']))
    progressBarProcess.start()

    # Start pool to work on jobs
    fingerprintingPool = Pool(CPU_COUNT)
    for file in fileList:
        jobQueue.put(file)
        fingerprintingPool.apply_async(getFingerprint)

    # Collect results
    for i in range(len(fileList)):
        result = resultsQueue.get()
        if result:
            rawData.append(result)

    fingerprintingPool.close()
    fingerprintingPool.join()
    progressBarProcess.join()

    # Calculate length of comparisionData (number of results at the end). ResultLen = 1 + 2 + 3 + ... + (len(rawData)-1)
    resultsLen = sum(range(len(rawData)))

    # Start new process to show progress over progress bar
    progressBarProcess = Process(target=progressBarMessenger, args=([resultsLen, resultsLen], ['Preparing fingerprints', 'Comparing fingerprints']))
    progressBarProcess.start()

    # Start new process to generate new jobs
    newJobsProcess = Process(target=newJobGenerator, args=([rawData]))
    newJobsProcess.start()

    # Start pool to work on jobs
    comparingPool = Pool(CPU_COUNT)
    for i in range(resultsLen):
        comparingPool.apply_async(compareFingerprints)

    # Collect results
    tqdm.write("Collecting results")
    for i in range(resultsLen):
        result = resultsQueue.get()
        comparisionData.append(result)

    comparingPool.close()
    comparingPool.join()
    progressBarProcess.join()
    newJobsProcess.join()

    # Sort results
    tqdm.write("Sorting results")
    sortedData = sorted(comparisionData, key=lambda x: x[2], reverse = True)

    # Norm Levenshtein scores from 0 to 100
    tqdm.write("Scaling Lavenshtein scores")
    minLavenshtein = min(sortedData, key=lambda x: x[3])[3]
    maxLavenshtein = max(sortedData, key=lambda x: x[3])[3]
    minNew = 100
    maxNew = 0

    for row in sortedData:
        row[3] = (((maxNew - minNew) / (maxLavenshtein - minLavenshtein)) * (row[3] - maxLavenshtein) + maxNew)

    # Save results
    tqdm.write("Saving results")
    with open(RESULTS_PATH, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['1st filename', '2nd filename', 'Fingerprint comparision result', 'Lavenshtein comparision result', 'Bigger score = Better match'])
        writer.writerows(sortedData)
        csvfile.close()

    tqdm.write('If you finished your analysis you can now delete temporary files in "' + PROGRESS_PATH + '" directory')
