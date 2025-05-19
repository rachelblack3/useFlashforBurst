import global_use as gl
import numpy as np


class FindLANLFeatures:

    """ Find the spacecraft positions at a given timestamp """

    def __init__(self,timestamp, LANL_epoch, MLT, MLAT_N, MLAT_S, Lstar):
    
        self.timestamp = timestamp
        self.LANL_data = LANL_epoch
        self.MLT = MLT
        self.MLAT_N = MLAT_N
        self.MLAT_S = MLAT_S
        self.Lstar = Lstar

    
    def index(self):

        LANL_epoch = self.LANL_data

        time, index = gl.find_closest(LANL_epoch, self.timestamp)

        return index

    @property 
    def get_MLAT(self):

        if np.isnan(self.MLAT_N[self.index()]) == True:

            MLAT = self.MLAT_S[self.index()]

        else:

            MLAT = self.MLAT_N[self.index()]

        return MLAT
    
    @property 
    def get_MLT(self):

        MLT = self.MLT[self.index()]

        return MLT

    @property 
    def get_Lstar(self):

        Lstar = self.Lstar[self.index()]

        return Lstar



class FindOMNIFeatures:

    def __init__(self,timestamp, OMNI_epoch, OMNI_epoch_low, AE, Kp, Dst):
    
        self.timestamp = timestamp
        self.OMNI_epoch = OMNI_epoch
        self.OMNI_epoch_low = OMNI_epoch_low
        self.AE = AE
        self.Kp = Kp
        self.Dst = Dst

    
    def index_high(self):

        OMNI_epoch = self.OMNI_epoch

        time, index = gl.find_closest(OMNI_epoch, self.timestamp)

        return index
    
    def index_low(self):

        OMNI_epoch = self.OMNI_epoch_low

        time,index = gl.find_closest(OMNI_epoch, self.timestamp)

        return index

    @property 
    def get_AE(self):

        AE = self.AE[self.index_high()]

        return AE
    
    @property 
    def get_Kp(self):

        Kp = self.Kp[self.index_low()]

        return Kp

    @property 
    def get_Dst(self):

        Dst = self.Dst[self.index_low()]

        return Dst      

class FindSurveyFeatures:

    """ Find the spacecraft positions at a given timestamp """

    def __init__(self,timestamp, epoch, ellipticity, planarity, wna,polar):
    
        self.timestamp = timestamp
        self.epoch = epoch
        self.ellipticity = ellipticity
        self.planarity = planarity
        self.wna = wna
        self.polar = polar
    
    def index(self):

        epoch = self.epoch

        time, index = gl.find_closest(epoch, self.timestamp)

        return index
    
    @property 
    def get_ellip(self):

        ellip = self.ellipticity[self.index(),:]

        return ellip  
    
    @property 
    def get_planar(self):

        planar = self.planarity[self.index(),:]

        return planar  
    
    @property 
    def get_wna(self):

        wna = self.wna[self.index(),:]

        return wna  
    
    @property 
    def get_polar(self):

        polar = self.polar[self.index(),:]

        return polar  
    


class FindLHR:

    """ Find the spacecraft positions at a given timestamp """

    def __init__(self,timestamp, epoch, flhr):
    
        self.timestamp = timestamp
        self.epoch = epoch
        self.flhr = flhr
    
    def index(self):

        epoch = self.epoch

        time, index = gl.find_closest(epoch, self.timestamp)

        return index

    @property 
    def get_lhr(self):


        LHR = self.flhr[self.index()]

        return LHR