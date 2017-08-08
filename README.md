# pbMoMa: Phase Based video MOtion MAgnification

A Python source code implementation of motion magnification based on the paper: [Phase Based Video Motion Processing](http://people.csail.mit.edu/mrub/papers/phasevid-siggraph13.pdf) by Neal Wadhwa, Michael Rubinstein, Frédo Durand, William T. Freeman, ACM Transactions on Graphics, Volume 32, Number 4 (Proc. SIGGRAPH), 2013. [project](http://people.csail.mit.edu/nwadhwa/phase-video/). 

#### Note: this follow up code can also handle large motion https://acceleration-magnification.github.io/


### Requirements:

 - python 2.7
 - numpy
 - [perceptual](https://github.com/andreydung/Steerable-filter) (Complex steerable pyramid, install with: sudo pip install perceptual) 

### Organization
 
    phasebasedMoMag.py      # Main file
    pyramid2arr.py          # Help class to convert a pyramid to a 1d array
    media/guitar.mp4        # Example video
     
### Example video

    ./media/guitar.mp4
    
When you run the code 'python phasebasedMoMag.py' it expects an example video in the 'media' folder. Here we use the [http://people.csail.mit.edu/mrub/evm/video/guitar.mp4](guitar.mp4) video from the motion magnification website.

 
### About

The pbMoMA implementation is based only on the paper. It was developed independent of the source code that can be requested from the paper authors (this pyton code was written without having access to that code). Therefore, the results from the pbMoMA code may differ from the results by the paper authors. Differences include: using a sliding window, only an Ideal filter, no sub-octave pyramid, and no color.
 
The code was implemented during the [Lorentz Center](http://www.lorentzcenter.nl/)  workshop [ICT with Industry: motion microscope](http://www.lorentzcenter.nl/lc/web/2015/775/info.php3?wsid=775&venue=Oort). Participants: Joao Bastos, Elsbeth van Dam, Coert van Gemeren, Jan van Gemert, Amogh Gudi, Julian Kooij, Malte Lorbach, Claudio Martella, Ronald Poppe.

