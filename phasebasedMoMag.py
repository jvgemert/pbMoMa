from perceptual.filterbank import *

import cv2
import cv2.cv as cv


import os, sys

import numpy as np

from pylab import *

import scipy.signal
import scipy.fftpack

from pyramid2arr import Pyramid2arr


def ideal_bandpass(data, fps, wl=1.5, wh=2.5, axis=0):
    fft = scipy.fftpack.fft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)    
    boundLow = (np.abs(frequencies - wl)).argmin()
    boundHigh = (np.abs(frequencies - wh)).argmin()
    fft[:boundLow] = 0
    fft[boundHigh:-boundHigh] = 0
    fft[-boundLow:] = 0
    frequencies[:boundLow] = 0
    frequencies[boundHigh:-boundHigh] = 0
    frequencies[-boundLow:] = 0
    
    return np.real( scipy.fftpack.ifft(fft, axis=axis) )

def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq): 

    # initialize the steerable complex pyramid
    steer = Steerable(5)

    print "Reading:", vidFname,

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))    
    width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))

    print width, height, 

    fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
    if np.isnan(fps):
        fps = 30
    print 'FPS:%d' % fps

    # video Writer
    fourcc = cv.CV_FOURCC('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    print 'Writing:', vidFnameOut

    # how many frames
    nrFrames = min(vidFrames, maxFrames)

    # read video
    #print steer.height, steer.nbands

    print 'FrameNr:', 
    for frameNr in range( nrFrames + windowSize ):
        print frameNr,
        sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
               
            if im == None:
                # if unexpected, quit
                break

            # convert to gray image
            grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)        

            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)

            # on first frame: init rotating array to store the pyramid coeffs 
            if frameNr ==0:
                # store a sliding window of frames, where the pyramid is converted to 1d arrays for easy filtering  
                pyArr = Pyramid2arr(steer, windowSize, coeff)
                
            # add image pyramid to video array
            pyArr.p2a(coeff)

        if frameNr >= windowSize-1:
            print '*',
            # get the phases of the window
            win_phases = pyArr.getPhases()
            
            # filter the window
            filteredPhases = ideal_bandpass(win_phases, fpsForBandPass, lowFreq, highFreq)
               
            # subtract the original response, add the magnified response
            frameIndex = max(frameNr - nrFrames -1, 0)
            
            magnifiedPhases = (win_phases[frameIndex,:] - filteredPhases[frameIndex,:]) + filteredPhases[frameIndex,:]*factor

            # creat coeffs     
            newCoeff = pyArr.a2p(phases=magnifiedPhases) 
            
            # reconstruct pyramid
            out = steer.reconSCFpyr(newCoeff)

            # clip values out of range
            out[out>255] = 255
            out[out<0] = 0
            
            # make a RGB image
            rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
            rgbIm[:,:,0] = out
            rgbIm[:,:,1] = out
            rgbIm[:,:,2] = out
            
            #write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release()   


################# main script

#vidFname = 'media/baby.mp4';
#vidFname = 'media/WIN_20151208_17_11_27_Pro.mp4.normalized.avi'
#vidFname = 'media/embryos01_30s.mp4'
vidFname = 'media/guitar.mp4'

# maximum nr of frames to process
maxFrames = 60000
# the size of the sliding window
windowSize = 30
# the magnifaction factor
factor = 20
# the fps used for the bandpass
fpsForBandPass = 600 # use -1 for input video fps
# low ideal filter
lowFreq = 72
# high ideal filter
highFreq = 92
# output video filename
vidFnameOut = vidFname + '-Mag%dIdeal-lo%d-hi%d.avi' % (factor, lowFreq, highFreq)

phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq)



