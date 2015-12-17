import numpy as np

class Pyramid2arr:
    '''Class for converting a pyramid to a sliding window of 1d arrays'''
    
    def __init__(self, steer, nrFrames, coeff):
        """
        Initialize class with sizes from pyramid coeff
        """
        self._frameNr = nrFrames
        self._steer = steer
        self._loPassArray = np.zeros((nrFrames, coeff[-1].shape[0], coeff[-1].shape[1]) )
        self._hiPassArray = np.zeros((nrFrames, coeff[0].shape[0], coeff[0].shape[1]) )
        self._sizes = []
        
        totSize = 0
        
        for lvl in range(1,self._steer.height-1):
            #print lvl, 'len(coeff[lvl])', len(coeff[lvl])
            for b in range(self._steer.nbands):
                totSize += coeff[lvl][b].size
                 
                self._sizes.append( coeff[lvl][b].shape ) 
                
                #print 'lvl', lvl, totSize
                #print self._sizes
        
        #self._bandArray = np.zeros((nrFrames, totSize), dtype='complex64')
        self._phases = np.zeros((nrFrames, totSize) )
        self._amplitudes = np.zeros((nrFrames, totSize) )

    def getPhases(self):
        """
        Return the phases of the sliding window as a (nrFrames, flattened pryamid) array
        """        
        start = self._phases.shape[0] - 1
        end = self._frameNr
        win = range(start,end,-1)
        return self._phases[win,:]


    def p2a(self, coeff):
        """
        Add pyramid as a 1d Array
        """
        
        # keep a window, 
        if self._frameNr > 0 :
            # keep adding if not yet filled up
            self._frameNr -= 1
        else:
            # if filled up, replace the oldest frame with the current
            self._hiPassArray = np.roll(self._hiPassArray, 1, 0)
            self._loPassArray = np.roll(self._loPassArray, 1, 0)
            self._phases = np.roll(self._phases, 1, 0)
            self._amplitudes = np.roll(self._amplitudes, 1, 0)
            
        
        self._hiPassArray[self._frameNr] = coeff[0]
        self._loPassArray[self._frameNr] = coeff[-1]
        
        bandArray = np.hstack([ np.ravel( coeff[lvl][b] ) for lvl in range(1,self._steer.height-1) for b in range(self._steer.nbands) ])
        
        self._phases[self._frameNr] = np.angle(bandArray)
        self._amplitudes[self._frameNr] = np.absolute(bandArray)
        
       
    def a2p(self, frameNr =-1, phases=[], amplitudes=[], loPassArray=[], hiPassArray=[] ):
        """
        Convert 1d array back to Pyramid
        """
        if frameNr == -1:
            frameNr = self._frameNr
        
        if phases == []:
            phases = self._phases[frameNr,:]
        if amplitudes == []:
            amplitudes = self._amplitudes[frameNr,:]
        if loPassArray == []:
            loPassArray = self._loPassArray[frameNr,:]
        if hiPassArray == []:
            hiPassArray = self._hiPassArray[frameNr,:]
    
            
        bandArray = amplitudes * np.exp(1j*phases)
        
        newCoeff = [ hiPassArray ]
        
        counter = 0
        sizeID = 0
        for lvl in range(1,self._steer.height-1):
            bandList = []
            for b in range(self._steer.nbands):
                size = self._sizes[sizeID]
                #print size
                # deal with 1d sizes
                if len(size) == 2:        
                    nr = size[0]*size[1]
                else:
                    nr = size[0]
                    
                arr = np.reshape( bandArray[counter:counter+nr], size )
                counter += nr
                sizeID += 1
                bandList.append( arr )
            #print 'len(bandList)', len(bandList)
            newCoeff.append( bandList )    
        
        newCoeff.append( loPassArray )
        return newCoeff

