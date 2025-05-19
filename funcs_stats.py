# general packages
import numpy as np
import pandas as pd

# plotting packages
import matplotlib.pyplot as plt

# global functions and constants
import global_use as gl

# general useful functions

def smooth(f, box_pts):
    ''' Function for smoothing functions using a moving average on f, of size box_pts'''
    box = np.ones(box_pts)/box_pts
    
    f_smooth = np.convolve(f, box, mode='same')
    return f_smooth

 # plot 6/8 in summary plot - frequency integrated Kletzing PSD

class FrequencyIntegral:

    def __init__(self,PSDs, lower_frequency, upper_frequency):

        self.PSD = PSDs["PSD"]
        self.Frequencies = PSDs["Frequencies"]
        self.Times = PSDs["Time"]
        self.lower = lower_frequency
        self.upper = upper_frequency

    def integrate_in_frequency(self):
        
        PSD, Frequencies = self.frequency_limits()
        n_f = len(Frequencies)
        n_bins = len(self.Times)

        frequency_integral = np.zeros(n_bins)
        
        for n in range(n_bins):
            bin_value = 0.

            # Integrate in frequency 
            for m in range(n_f-1):
                    
                bin_value += 0.5*(PSD[n,m] + PSD[n,m+1])*(Frequencies[m+1]-Frequencies[m])

            frequency_integral[n] = bin_value

        max_amplitude = np.max(np.sqrt(frequency_integral))

        integral_statistics = {"maximum amplitude": max_amplitude,
                            "mean power": np.mean(frequency_integral),
                            "median power": np.median(frequency_integral),
                            "power vairaince": np.std(frequency_integral)**2,
                            "frequency integrated power": frequency_integral}

        return integral_statistics
    

    def frequency_limits(self):
        ''' Finding upper and lower boundaries of PSD from gievn frequency limits (using linear interpolation )'''

        # for lower frequency limit
        for i in range(len(self.Frequencies)):

            if self.Frequencies[i] > self.lower: 
                
                # Do linear interpolation
                x_num = self.PSD[:,i] - self.PSD[:,i-1]
                y_num = self.Frequencies[i]- self.Frequencies[i-1]
                PSD_lower = self.PSD[:,i-1] + (self.lower - self.Frequencies[i-1])*(x_num/y_num)
                
                # index after new lower frequency limit
                lower_index = i

                # now found, exit loop
                break

        # for upper frequency limit
        for i in range(len(self.Frequencies)):

            if self.Frequencies[i] > self.upper: 

                # do linear interpolation
                x_num = self.PSD[:,i] - self.PSD[:,i-1]
                y_num = self.Frequencies[i]- self.Frequencies[i-1]
                PSD_upper = self.PSD[:,i-1] + (self.upper - self.Frequencies[i-1])*(x_num/y_num)
                
                # index just before the new upper frequency limit
                upper_index = i - 1
                
                # now found, exit loop
                break

        # Make sure 'array' of new column values for lower and upper interpolated freqeuncies are 2D
        PSD_lower = PSD_lower.reshape(len(self.Times),1)
        PSD_upper = PSD_upper.reshape(len(self.Times),1)

        # stich together the desired range of original PSD array and the the inteprolated lower and upper values
        PSD_desired = np.hstack((PSD_lower, self.PSD[:,lower_index:upper_index]))
        PSD_desired = np.hstack((PSD_desired, PSD_upper))

        # similarly, stick together the new desired frequency range
        Frequencies_desired = self.Frequencies[lower_index:upper_index]
        Frequencies_desired = np.insert(Frequencies_desired, 0, self.lower)
        Frequencies_desired = np.append(Frequencies_desired, self.upper)


        return PSD_desired, Frequencies_desired
    
class TimeIntegral:

    def __init__(self,PSD,Frequency,Times): 

        self.Times = Times
        self.PSD = PSD
        self.Frequencies = Frequency
    
    def integrate_in_time(self):
        
        n_t = len(self.Times)-1
        n_f = len(self.Frequencies)

        time_integral = np.zeros(n_f)

        for m in range(n_f):
            bin_value = 0.
            # Integrate in time 
            for n in range(n_t-1):
                    
                bin_value += 0.5*(self.PSD[n,m] + self.PSD[n+1,m])*(self.Times[n+1] - self.Times[n])

            time_integral[m] = bin_value

        return time_integral
        


class Rebin:

    def __init__(self, PSD, Frequency, bins):

        self.PSD = PSD
        self.Frequencies_have = Frequency
        self.Frequency_bin = bins

    def do_rebin(self):

        """ 
        Read in semi-log bins and define bin edges for rebinning
        """

        """ 
        Doing the rebinning
        """
        PSD = np.array(self.PSD).T

        # Create dataframe
        rebin_dat=pd.DataFrame(PSD)


        #rebin_dat['Data'] = PSD
        
        # Create and save frequencies to one column
        rebin_dat['Frequency']= self.Frequencies_have
        
        """
        pd.cut() bins all frequencies according to defined semi_log bins
        groupby() groups all data in frame by these bines
        then select the DATA and take the MEAN in each bin
        to_numpy saves this to an array
        """
        
        rebinned=rebin_dat.groupby(pd.cut(rebin_dat.Frequency,bins=self.Frequency_bin)).mean().to_numpy()

        return rebinned[:,:-1]

    

class PlotDist:

    def __init__(self, FFT, integral_stats, flag, axs):

        self.FFT = FFT
        self.flag = flag
        self.integral_stats = integral_stats
        self.ax = axs

    def plot_full(self):
        ''' 
            This plots the FFT frequency integrated power between:
            1. 0.05 fce and 0.5 fce (dashed)
            2. 0.5 fce and 1 fce (dashed)
            3. 0.05 fce and 1 fce (filled) 
        '''
        ax = self.ax

        # power integral
        ax.plot(self.FFT["Time"],self.integral_stats["frequency integrated power"], label = f"0.05 - 0.5 fce",linestyle = 'dashed', marker = '*')
    
        # also plot mean/median
        ax.axhline(self.integral_stats["median power"],np.min(self.FFT["Time"]), np.max(self.FFT["Time"]), color = 'b', label = 'Median')
        ax.axhline(self.integral_stats["median power"],np.min(self.FFT["Time"]), np.max(self.FFT["Time"]), color = 'lightblue', label = 'Mean')

        # set the limits and labels
        ax.set_xlim(np.min(self.FFT["Time"]), np.max(self.FFT["Time"]))
        ax.set_xlabel(r'$ Time\ (s)$')
        ax.set_ylabel(r'$ Frequency\ integrated\ power\ (nT^2)$')

        # plot details
        ax.set_yscale("log")
        ax.legend()


class PlotStep:

    def __init__(self,intK,intS,survey_value,ax):

        self.intK = intK
        self.intS = intS
        self.survey_value = survey_value
        self.ax = ax

    def plot(self):


        ax = self.ax

        steps, steps_time = SlidingSteps(self.intS).make_steps()
        print("Sliding steps:", steps)
        print(steps_time)
        ax.plot(steps_time,steps,label=r'$0.468s\ sliding-window\ integrated\ power\ averages$', color = 'lightblue')

        steps, steps_time = KletzingSteps(self.intK).make_steps()
        print(steps_time)
        print("kletzing steps:", steps)          
        ax.plot(steps_time,steps,label=r'$Kletzing-window\ integrated\ power$', color = 'darkblue')
        ax.axhline(self.survey_value,0,steps_time[-1],color='black',label=r'$Integrated\ survey\ power$')
        ax.set_xlim(0,5.968)
        ax.set_xlabel(r'$Time\ (s)$')
        ax.set_ylabel(r'$Power\ (nT)$')
        ax.legend()


class SlidingSteps:

    "Taking average of sliding windows every 0.468s to create step plot"
     
    def __init__(self, integral_sliding):
          
        self.intS = integral_sliding
        self.N = gl.global_constants["N Kletzing"]
    
    def make_steps(self):
        
        average = self.average_over_0468()
        # Create empty lists
        values = np.zeros(2*self.N)
        value_times = np.zeros_like(values)

        duration = gl.global_constants["Duration"]
        Twin = gl.global_constants["Kletzing box"]*gl.global_constants["f_s"]
        
        times = np.linspace(Twin, Twin*self.N, self.N)

        value_times[0] = 0 
        value_times[-1] = times[-1]

        for i in range(self.N):
            values[2*i] = average[i]
            values[2*i + 1] = average[i]

            if i<11:
                value_times[2*i + 1] = times[i]
                value_times[2*i + 2] = times[i]


        return values, value_times
    
    def average_over_0468(self):

        # total no time bins
        n_t = len(self.intS)
        # initialise array for average from sliding windows for each of the 12 Kletzing windows
        average_dist = np.zeros(self.N)
        # number of temporal bins in each Kletzing window
        #n_0468 = int((gl.global_constants["Kletzing box"]/gl.global_constants["N total"])*n_t)
        n_0468 = int((gl.global_constants["Kletzing box"]/gl.global_constants["Sliding box"]))
        print("The time is over which the 0.03s is taken is",n_0468*gl.global_constants["f_s"]*gl.global_constants["Sliding box"])

        times = []

        # do averaging
        for i in range(self.N):
            average_dist[i] = np.mean(self.intS[i*n_0468:(i+1)*n_0468])
            print(i*n_0468)
            times.append(i*n_0468*gl.global_constants["f_s"]*gl.global_constants["Sliding box"])

        times.append(12*n_0468*gl.global_constants["f_s"]*gl.global_constants["Sliding box"])

        return average_dist
    

class KletzingSteps:

    def __init__(self, integral_kletzing):
          
        self.intK = integral_kletzing
        self.N = gl.global_constants["N Kletzing"]
    
    def make_steps(self):
        
        average = self.intK
        # Create empty lists
        values = np.zeros(2*self.N)
        value_times = np.zeros_like(values)

        duration = gl.global_constants["Duration K"]
        Twin = gl.global_constants["Kletzing box"]*gl.global_constants["f_s"]
        
        times = np.linspace(Twin, Twin*self.N, self.N)

        value_times[0] = 0 
        value_times[-1] = times[-1]

        for i in range(self.N):
            values[2*i] = average[i]
            values[2*i + 1] = average[i]

            if i<11:
                value_times[2*i + 1] = times[i]
                value_times[2*i + 2] = times[i]

        return values, value_times



class FindPeaks:

    def __init__(self, frequency_spectrum):

        self.spectrum = frequency_spectrum




class FrequencyIntegralFull:

    def __init__(self,PSD,PSD_features,frequency_required,lower_limit,fce,binwidth):

        self.PSD = PSD
        self.Frequencies = PSD_features["Frequencies"]
        self.Times = PSD_features["Time"]
        self.lower_limit = lower_limit
        self.desired_frequencies = frequency_required
        self.fce = fce
        self.binwidth = binwidth

    def integrate_in_frequency(self):
        
        PSD = self.PSD
        Frequencies = self.Frequencies

        # rebin the Kletzing FFTs to be in the same frequency bins as the 0.03s FFT
        
        if len(Frequencies) != len(self.desired_frequencies):
            bin_edges = list(self.desired_frequencies)
            bin_edges.append(bin_edges[-1]+bin_edges[1])

            PSD = Rebin(PSD, Frequencies, bin_edges).do_rebin().T

        n_bins = np.shape(PSD)[0]           # no. of temporal bins in PSD
        n_01 = 10                           # no. of 0.1 fce frequency bands between flhr and fce
        
        # Create array for stroing all frequency integrals - dimensions (no. of 0.1 fce bands, no. of temporal bins in PSD)
        frequency_integral = np.zeros((n_01,n_bins))

        # get list of frequency bounds on 01 fce bands
        f_01 = self.frequency_bands()

        # initialise arrays for stats: max amplitude, mean power, median power, std in the power
        max_amp = np.zeros(n_01)
        mean_pow = np.zeros(n_01)
        med_pow = np.zeros(n_01)
        std_pow = np.zeros(n_01)


        for k in range(n_01-1):

            f_l = f_01[k]
            f_u=f_01[k+1]

            # if we go over the upper lim of the FFT frequencies, set all the integrals to 0
            if (f_u>Frequencies[-1]) or (f_l>Frequencies[-1]):
                mask = np.zeros_like(max_amp, dtype=bool)
                mask[k:] = True
                # save stats for each band
                max_amp[mask] = np.nan
                mean_pow[mask] = np.nan
                med_pow[mask] = np.nan
                std_pow[mask] = np.nan

                mask = np.zeros_like(frequency_integral, dtype=bool)
                mask[k:,:] = True
                frequency_integral[mask] = np.nan
                break
            
            # but if we are within correct range...
            else:
                # find interpolated PSD in frequency range
                PSD_band, freq_band = self.inteprolate_PSD(f_l,f_u)
                n_f = len(freq_band)
                


                for n in range(n_bins):
                    bin_value = 0.
                    # Integrate in frequency 
                    for m in range(n_f-1):
                    
                        bin_value += 0.5*(PSD_band[n,m] + PSD_band[n,m+1])*(freq_band[m+1]-freq_band[m])

                    frequency_integral[k,n] = bin_value

                # save stats for each band
                max_amp[k] = np.nanmax(frequency_integral[k,:])
                mean_pow[k] = np.nanmean(frequency_integral[k,:])
                med_pow[k] = np.nanmedian(frequency_integral[k,:])
                std_pow[k] = np.nanstd(frequency_integral[k,:])


        integral_statistics = {"maximum amplitude": max_amp,
                            "mean power": mean_pow,
                            "median power": med_pow,
                            "power std": std_pow,
                            "frequency integrated power": frequency_integral}
    
      
        return integral_statistics

    def frequency_bands(self):

        # Find the frequencies corresponding to integer 0.1 fce bands
        # will need to do interpolations between bins for this...

        f_01 = []

        # append f_lhr as the lowest frequency band limit
        f_01.append(self.lower_limit)

        for i in range(1,10):
            f_01.append(i*0.1*self.fce)

        # return list of frequencies
        return f_01
    
    def inteprolate_PSD(self, f_l, f_u):
        
        # for power fraction in bin at lower frequency limit
        for i in range(len(self.Frequencies)):
            if self.Frequencies[i]-self.binwidth/2 <= f_l < self.Frequencies[i]+self.binwidth/2:
                print("i'm here - lower lim")
                index_l = i+1
                print("the index is", index_l)
                power_below_frac = (self.Frequencies[i]+self.binwidth/2 - f_l)/self.binwidth
                PSD_frac_below = self.PSD[:,i]*power_below_frac
                PSD_frac_below = PSD_frac_below.reshape(len(self.Times)-1,1)
                break
            else:
                index_l = 0
                                                        
        # for power fraction in bin at upper frequency limit
        for i in range(len(self.Frequencies)):
            if self.Frequencies[i]-self.binwidth/2 <= f_u < self.Frequencies[i]+self.binwidth/2:
                print("i'm here - upper lim")
                index_u = i-1
                print("the index is", index_u, "but the last freq is", len(self.Frequencies))
                power_above_frac = 1 - (self.Frequencies[i]+self.binwidth/2 - f_u)/self.binwidth
                PSD_frac_above= self.PSD[:,i]*power_above_frac
                PSD_frac_above= PSD_frac_above.reshape(len(self.Times)-1,1)
                break
            else:
                index_u = -1
        
        # Now combine all together for integration range
        if (index_l == 0) & (index_u == -1):

            PSD_desired = self.PSD
            Frequencies_desired = self.Frequencies[index_l:index_u+1]
            print(np.shape(PSD_desired),np.shape(Frequencies_desired))

        elif (index_l == 0):

            PSD_desired = np.hstack((self.PSD[:,index_l:index_u+1],PSD_frac_above))
            print(np.shape(PSD_desired))
            Frequencies_desired = self.Frequencies[index_l:index_u]
            Frequencies_desired = np.append(Frequencies_desired, f_u)
            print(np.shape(PSD_desired),np.shape(Frequencies_desired))

        elif (index_u == -1):

            PSD_desired = np.hstack((PSD_frac_below,self.PSD[:,index_l:index_u]))
            Frequencies_desired = self.Frequencies[index_l:index_u]
            Frequencies_desired = np.insert(Frequencies_desired, 0, f_l)
            print(np.shape(PSD_desired),np.shape(Frequencies_desired))

        elif (index_u==index_l):
            PSD_desired=np.zeros_like(self.PSD)
            Frequencies_desired=np.zeros_like(self.Frequencies)

        else:
            print("prob here - 4")
            print(np.shape(self.PSD),index_l,index_u)
            # Now combine all together for integration range
            PSD_desired = np.hstack((PSD_frac_below,self.PSD[:,index_l:index_u],PSD_frac_above))

        # similarly, stick together the new desired frequency range
            Frequencies_desired = self.Frequencies[index_l:index_u]
            print('the indices:',index_l, index_u)
            Frequencies_desired = np.insert(Frequencies_desired, 0, f_l)
            Frequencies_desired = np.append(Frequencies_desired, f_u)
            print(np.shape(PSD_desired),np.shape(Frequencies_desired))

        return PSD_desired, Frequencies_desired

class FrequencyIntegralSurvey:

    def __init__(self,PSDs, timestamp, f_lhr, fce, binwidths):

        self.PSD = PSDs["Bmagnitude"]
        self.Frequencies = PSDs["Frequency"]
        self.Times = PSDs["Epoch"]
        self.timestamp = timestamp
        self.flhr = f_lhr
        self.fce = fce
        self.binwidth = binwidths

    def integrate_in_frequency(self):
        
        PSD = self.PSD
        Frequencies = self.Frequencies

        n_bins = np.shape(PSD)[0]           # no. of temporal bins in PSD
        n_01 = 10                           # no. of 0.1 fce frequency bands between flhr and fce
        
        # Create array for stroing all frequency integrals - dimensions (no. of 0.1 fce bands, no. of temporal bins in PSD)
        frequency_integral = np.zeros(n_01)

        # get list of frequency bounds on 01 fce bands
        f_01 = self.frequency_bands()

        for n in range(n_bins):

            if self.Times[n] == self.timestamp.replace(microsecond=0):
                print('survey time found:', self.timestamp)

                for k in range(n_01-1):

                    f_l = f_01[k]
                    f_u=f_01[k+1]

                    # if we go over the upper lim of the FFT frequencies, set all the integrals to 0
                    if f_u>Frequencies[-1]:

                        mask = np.zeros_like(frequency_integral, dtype=bool)
                        mask[k:] = True
                        frequency_integral[mask] = np.nan
                        break
                    
                    # but if we are within correct range...
                    else:

                        # find interpolated PSD in frequency range
                        PSD_band, freq_band = self.inteprolate_PSD(f_l,f_u)
                        n_f = len(freq_band)
                    
                        # Integrate in frequency 
                        for m in range(n_f-1):

                            frequency_integral[k] += 0.5*(PSD_band[n,m] + PSD_band[n,m+1])*(freq_band[m+1]-freq_band[m])

    
        print("The frequency integral for the survey is:",frequency_integral)
        
        return frequency_integral, f_01
    
    def frequency_bands(self):

        # Find the frequencies corresponding to integer 0.1 fce bands
        # will need to do interpolations between bins for this...

        f_01 = []

        # append f_lhr as the lowest frequency band limit
        f_01.append(self.flhr)

        for i in range(1,10):
            f_01.append(i*0.1*self.fce)

        # return list of frequencies
        return f_01

    def inteprolate_PSD(self, f_l, f_u):
        
        # for power fraction in bin at lower frequency limit
        for i in range(len(self.Frequencies)):
            if self.Frequencies[i]-self.binwidth[i]/2 <= f_l < self.Frequencies[i]+self.binwidth[i]/2:
                index_l = i+1
                power_below_frac = (self.Frequencies[i]+self.binwidth[i]/2 - f_l)/self.binwidth[i]
                PSD_frac_below = self.PSD[:,i]*power_below_frac
                PSD_frac_below = PSD_frac_below.reshape(len(self.Times),1)
                break
            else:
                index_l = 0
                                                        
        # for power fraction in bin at upper frequency limit
        for i in range(len(self.Frequencies)):
            if self.Frequencies[i]-self.binwidth[i]/2 <= f_u < self.Frequencies[i]+self.binwidth[i]/2:
                index_u = i-1
                power_above_frac = 1 - (self.Frequencies[i]+self.binwidth[i]/2 - f_u)/self.binwidth[i]
                PSD_frac_above= self.PSD[:,i]*power_above_frac
                PSD_frac_above= PSD_frac_above.reshape(len(self.Times),1)
                break
            else:
                index_u = -1

          # Now combine all together for integration range
        if (index_l == 0) & (index_u == -1):
            PSD_desired = self.PSD
                # similarly, stick together the new desired frequency range
            Frequencies_desired = self.Frequencies
            
        elif (index_l == 0):
            PSD_desired = np.hstack((self.PSD[:,index_l:index_u],PSD_frac_above))
            # similarly, stick together the new desired frequency range
            Frequencies_desired = self.Frequencies[index_l:index_u]
            Frequencies_desired = np.append(Frequencies_desired, f_u)

        elif (index_u == -1):
            PSD_desired = np.hstack((PSD_frac_below,self.PSD[:,index_l:index_u]))
            # similarly, stick together the new desired frequency range
            Frequencies_desired = self.Frequencies[index_l:index_u]
            Frequencies_desired = np.insert(Frequencies_desired, 0, f_l)
            

        else:
            # Now combine all together for integration range
            PSD_desired = np.hstack((PSD_frac_below,self.PSD[:,index_l:index_u],PSD_frac_above))

            # similarly, stick together the new desired frequency range
            Frequencies_desired = self.Frequencies[index_l:index_u]
            Frequencies_desired = np.insert(Frequencies_desired, 0, f_l)
            Frequencies_desired = np.append(Frequencies_desired, f_u)

        return PSD_desired, Frequencies_desired
    


class Plasmapause:

    def __init__(self, timestamp, fpe_in, fpe_out, fpe_epoch):

        self.timestamp = timestamp
        self.fpe_in = fpe_in
        self.fpe_out = fpe_out
        self.epoch = fpe_epoch

    
    def in_or_out(self):

        time, index = gl.find_closest(self.epoch, self.timestamp)

        if np.isnan(self.fpe_out[index]) == True:

            flag = 'In'

        else:

            flag = 'Out'

        return flag
    
