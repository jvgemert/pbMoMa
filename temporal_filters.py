# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:52:29 2015

@author: jkooij
"""

import numpy as np
import scipy.signal

import scipy.fftpack as fftpack
#import pyfftw.interfaces.scipy_fftpack as fftpack


class SlidingWindow (object):
    
    def __init__(self, size, step=1):
        self.size = size
        self.step = step
        self.memory = None
        
        assert(self.step > 0)
    
    def process(self, data_itr):
        """ Generator for windows after giving it more data.
        
            Example:
            
            winsize = 2
            win = SlidingWindow(winsize)
            batches = (np.random.randint(0,9, 3) for _ in range(3))
            for w in win.process(batches):
                print '<<<', w        
        """
        for data in data_itr:
            self.update(data)
            while True:
                try:
                    out = self.next()
                    yield out
                except StopIteration:
                    break
    
    def update(self, data):
        if self.memory is None:
            self.memory = np.asarray(data)
        else:
            self.memory = np.concatenate((self.memory, data), axis=0)
        
    def next(self):
        if self.memory is not None and self.memory.shape[0] >= self.size:
            # get window
            out = self.memory[:self.size]
            
            # slide
            self.memory = self.memory[self.step:]
            
            return out
        else:
            raise StopIteration()
    
    def collect(self):
        # collect remainder of sliding windows
        out = []
        while True:
            try:
                out.append(self.next())
            except StopIteration:
                break
        return np.array(out)

class IdealFilter (object):
    """ Implements ideal_bandpassing as in EVM_MAtlab. """
    
    def __init__(self, wl=.5, wh=.75, fps=1, NFFT=None):
        """Ideal bandpass filter using FFT """

        self.fps = fps
        self.wl = wl
        self.wh = wh
        self.NFFT = NFFT
        
        if self.NFFT is not None:
            self.__set_mask()
            
    def __set_mask(self):
        self.frequencies = fftpack.fftfreq(self.NFFT, d=1.0/self.fps)    
        
        # determine what indices in Fourier transform should be set to 0
        self.mask = (np.abs(self.frequencies) < self.wl) | (np.abs(self.frequencies) > self.wh)
        

    def __call__(self, data, axis=0):
        if self.NFFT is None:
            self.NFFT = data.shape[0]
            self.__set_mask()            
            
        fft = fftpack.fft(data, axis=axis)        
        fft[self.mask] = 0   
        return np.real( fftpack.ifft(fft, axis=axis) )        

class IdealFilterWindowed (SlidingWindow):
    
    def __init__(self, winsize, wl=.5, wh=.75, fps=1, step=1, outfun=None):
        SlidingWindow.__init__(self, winsize, step)
        self.filter = IdealFilter(wl, wh, fps=fps, NFFT=winsize)
        self.outfun = outfun
        
    def next(self):
        out = SlidingWindow.next(self)
        out = self.filter(out)
        if self.outfun is not None:
            # apply output function, e.g. to return first (most recent) item
            out = self.outfun(out)
        return out


class IIRFilter (SlidingWindow):
    """ 
    Implements the IIR filter
           a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
                                   - a[1]*y[n-1] - ... - a[na]*y[n-na]        
    See scipy.signal.lfilter
    """

    def __init__(self, b, a):
        
        self.b = b
        self.a = a
        self.nb = len(b)
        self.na = len(a)
        
        # put parameters in right order for calculation
        #  (i.e. parameter of most recent time step last)
        self.b_ = b[::-1]
        self.a_ = a[-1:0:-1] # exclude a[0], it's used to scale output
        
        # setup sliding windows for input x and output y
        self.windowy = SlidingWindow(self.na-1)
        SlidingWindow.__init__(self, self.nb)
        
    def update(self, data):
        if self.memory is None:
            # prepend zeros
            data = np.asarray(data)
            zsize = (self.nb-1,) + data.shape[1:]
            data = np.concatenate((np.zeros(zsize), data), axis=0)
            
            # initialize output memory with zerostoo
            zsize = (self.na-1,) + data.shape[1:]
            self.windowy.update(np.zeros(zsize))
            
        SlidingWindow.update(self, data)

    def next(self):
        winx = SlidingWindow.next(self)
        winy = self.windowy.next()
        y = np.dot(self.b_, winx) - np.dot(self.a_, winy)

        self.windowy.update([y])
            
        return y / self.a[0]

        
class ButterFilter (IIRFilter):
    def __init__(self, n, freq, fps=1, btype='low'):
        freq = float(freq) / fps
        (b,a) = scipy.signal.butter(n, freq, btype)
        IIRFilter.__init__(self, b, a)


class ButterBandpassFilter (ButterFilter):
    
    def __init__(self, n, freq_low=.25, freq_high=.5, fps=1):
        ButterFilter.__init__(self, n, freq_high, fps=fps, btype='low')

        # additional low-pass
        self.lowpass = ButterFilter(n, freq_low, fps=fps, btype='low')
        
    def update(self, data):
        ButterFilter.update(self, data)
        self.lowpass.update(data)
    
    def next(self):
        out = ButterFilter.next(self)
        out_low = self.lowpass.next()
        return (out - out_low)
