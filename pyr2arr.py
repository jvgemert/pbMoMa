# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:19:12 2015

@author: jkooij
"""

import numpy as np

class Pyramid2arr:
    '''Class for converting a pyramid to/from a 1d array'''
    
    def __init__(self, steer, coeff=None):
        """
        Initialize class with sizes from pyramid coeff
        """
        self.levels = range(1, steer.height-1)
        self.bands = range(steer.nbands)
        
        self._indices = None
        if coeff is not None:
            self.init_coeff(coeff)

    def init_coeff(self, coeff):       
        shapes = [coeff[0].shape]        
        for lvl in self.levels:
            for b in self.bands:
                shapes.append( coeff[lvl][b].shape )             
        shapes.append(coeff[-1].shape)

        # compute the total sizes        
        sizes = [np.prod(shape) for shape in shapes]
        
        # precompute indices of each band
        offsets = np.cumsum([0] + sizes)
        self._indices = zip(offsets[:-1], offsets[1:], shapes)

    def p2a(self, coeff):
        """
        Convert pyramid as a 1d Array
        """
        
        if self._indices is None:
            self.init_coeff(coeff)
        
        bandArray = np.hstack([ np.ravel( coeff[lvl][b] ) for lvl in self.levels for b in self.bands ])
        bandArray = np.hstack((np.ravel(coeff[0]), bandArray, np.ravel(coeff[-1])))

        return bandArray        
        
       
    def a2p(self, bandArray):
        """
        Convert 1d array back to Pyramid
        """
        
        assert self._indices is not None, 'Initialize Pyramid2arr first with init_coeff() or p2a()'

        # create iterator that convert array to images
        it = (np.reshape(bandArray[istart:iend], size) for (istart,iend,size) in self._indices)
        
        coeffs = [it.next()]
        for lvl in self.levels:
            coeffs.append([it.next() for band in self.bands])
        coeffs.append(it.next())

        return coeffs

