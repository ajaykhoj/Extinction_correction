import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# this code will correct the extinction and reddening correction of magnitude of source and convert into flux density

### B filter
# a, b = 0.9994, 1.0171
# Z, dZ = 19.11, 0.02
# Conversion factor = 1.472e-16
# Wavelength in (angstr.) = 4392

### U filter
# a, b = 0.9226, 2.1019
# Z, dZ = 18.34, 0.02
# Conversion factor = 1.628e-16
# Wavelength in (angstr.) = 3465

### V filter
# a, b = 1.0015, 0.0126
# Z, dZ = 17.89, 0.01
# Conversion factor = 2.613e-16
# Wavelength in (angstr.) = 5468

### M2 filter
# a, b = 0.0773, 9.1784
# Z, dZ = 16.82, 0.03
# Conversion factor = 8.446e-16
# Wavelength in (angstr.) = 2246

### W1 filter
# a, b = 0.4346, 5.3286
# Z, dZ = 17.44, 0.03
# Conversion factor = 4.209e-16
# Wavelength in (angstr.) = 2600

### W2 filter
# a, b = -0.0581, 8.4402
# Z, dZ = 17.38, 0.03
# Conversion factor = 5.976e-16
# Wavelength in (angstr.) = 1928

### Extinction (A_lambda) = E_(B-V) * (a * Rv + b)
### Zpt = m_source + 2.5log(C_source), where Zpt is zero point mag., C_source is count rate
### C_source = 10**(0.4 * (Zpt - m_source_corr))
### Flux_density = C_source * conversion_factor (ergs/cm2/sec/A)

class ExtinctionOld:
    @staticmethod
    def extinction_B(Ebv, mag, dmag):
        A = Ebv * (0.9994*3.1 + 1.0171)
        mag_corr = mag - A                                               # corrected magnitude 
        Csource = 10 ** (0.4 * (19.11 - mag_corr))                       # Count rate 
        F_corr = Csource * 1.472e-16                                     # flux density (ergs/cm**2/sec/Angs.)
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.02**2) # flux density error (ergs/cm**2/sec/Angs.)
        Flux = 4392 * F_corr                                             # flux (ergs/cm**2/sec)
        dFlux = 4392 * dF_corr                                           # flux error (ergs/cm**2/sec)
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_U(Ebv, mag, dmag):
        A = Ebv * (0.9226 * 3.1 + 2.1019)
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (18.34 - mag_corr))
        F_corr = Csource * 1.628e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.02**2)
        Flux = 3465 * F_corr
        dFlux = 3465 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_V(Ebv, mag, dmag):
        A = Ebv * (1.0015 * 3.1 + 0.0126)
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.89 - mag_corr))
        F_corr = Csource * 2.613e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.01**2)
        Flux = 5468 * F_corr
        dFlux = 5468 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_M2(Ebv, mag, dmag):
        A = Ebv * (0.0773 * 3.1 + 9.1784)
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (16.82 - mag_corr))
        F_corr = Csource * 8.446e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 2246 * F_corr
        dFlux = 2246 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_W1(Ebv, mag, dmag):
        A = Ebv * (0.4346 * 3.1 + 5.3286)
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.44 - mag_corr))
        F_corr = Csource * 4.209e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 2600 * F_corr
        dFlux = 2600 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_W2(Ebv, mag, dmag):
        A = Ebv * (-0.0581 * 3.1 + 8.4402)
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.38 - mag_corr))
        F_corr = Csource * 5.976e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 1928 * F_corr
        dFlux = 1928 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

class ExtinctionNew:
    @staticmethod
    def extinction_B1(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (19.11 - mag_corr))
        F_corr = Csource * 1.472e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.02**2)
        Flux = 4392 * F_corr
        dFlux = 4392 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_U1(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (18.34 - mag_corr))
        F_corr = Csource * 1.628e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.02**2)
        Flux = 3465 * F_corr
        dFlux = 3465 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_V1(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.89 - mag_corr))
        F_corr = Csource * 2.613e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.01**2)
        Flux = 5468 * F_corr
        dFlux = 5468 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_M21(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (16.82 - mag_corr))
        F_corr = Csource * 8.446e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 2246 * F_corr
        dFlux = 2246 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_W11(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.44 - mag_corr))
        F_corr = Csource * 4.209e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 2600 * F_corr
        dFlux = 2600 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux

    @staticmethod
    def extinction_W21(mag, dmag, A):
        mag_corr = mag - A
        Csource = 10 ** (0.4 * (17.38 - mag_corr))
        F_corr = Csource * 5.976e-16
        dF_corr = F_corr * np.log(10) * 0.4 * np.sqrt(dmag**2 + 0.03**2)
        Flux = 1928 * F_corr
        dFlux = 1928 * dF_corr
        return mag_corr, Csource, F_corr, dF_corr, Flux, dFlux
    
    
############################################# Extinction correction of UVOT lightcurve 
class ExtinctionCorrectionUVOTLC:
    def __init__(self, time, mag, magerr, method="Old", filter_name="B", A=0, Ebv=0):
        """
        read the data

        """
        self.time = time
        self.mag = mag
        self.magerr = magerr
        self.method = method
        self.filter_name = filter_name
        self.A = A
        self.Ebv = Ebv

    def perform_correction(self):
        """
        Perform extinction correction based on the chosen method and filter.
        
        return: A list of tuples containing corrected data.
        """
        results = []

        methods = {
            "Old": {
                "B": ExtinctionOld.extinction_B,
                "U": ExtinctionOld.extinction_U,
                "V": ExtinctionOld.extinction_V,
                "M2": ExtinctionOld.extinction_M2,
                "W1": ExtinctionOld.extinction_W1,
                "W2": ExtinctionOld.extinction_W2,
            },
            "New": {
                "B": ExtinctionNew.extinction_B1,
                "U": ExtinctionNew.extinction_U1,
                "V": ExtinctionNew.extinction_V1,
                "M2": ExtinctionNew.extinction_M21,
                "W1": ExtinctionNew.extinction_W11,
                "W2": ExtinctionNew.extinction_W21,
            },
        }

        if self.filter_name not in methods[self.method]:
            raise ValueError(f"Invalid filter: {self.filter_name}")

        correction_function = methods[self.method][self.filter_name]

        for t, mag, dmag in zip(self.time, self.mag, self.magerr):
            if self.method == "Old":
                result = correction_function(self.Ebv, mag, dmag)
                print('Working with Old code')
            else:
                result = correction_function(mag, dmag, self.A)
                print('Working with New code')
                
            results.append((t, *result))

        return results

    def save_results(self, results, output_file="UVOT_corrected_lightcurve"):
        """
        Save the corrected results to a file.
        return: a output file contains " time,
                                         corrected magnitude,
                                         Counts,
                                         corrected flux_density,
                                         corrected flux_density error,
                                         corrected Flux,
                                         corrected Flux error
        """
        self.filename = f"{output_file}.txt"
        
        with open(self.filename, "w") as f:
            f.write("Time Corrected_Mag Count_Rate Flux_Density eFlux_Density Flux eFlux\n")
            for res in results:
                f.write(f"{res[0]} {' '.join(map(str, res[1:]))}\n")
        print(f"Results saved to {output_file}")

    def plot_results(self, results):
        """
        Plot the corrected flux with error bars as a function of time (lightcurve).
        """
        
        Time = [res[0] for res in results]
        fluxcorr = [res[5] for res in results]
        errorcorr = [res[6] for res in results]
        print(Time,fluxcorr,errorcorr)

        plt.figure(figsize=(11, 7), dpi=800)
        plt.errorbar(Time, fluxcorr, yerr=errorcorr, fmt="o", color="k", ecolor="red", alpha=0.7, capsize=3)
        plt.ylabel(r'$\nu F_{\nu} \ (\mathrm{erg} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1})$', fontsize=16)
        plt.xlabel("Time (MJD)", fontsize=16)
        plt.title("Corrected Light Curve", fontsize=18)
        #plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

          
############################################# Extinction correction of UVOT SED 
class ExtinctionCorrectionUVOTSED:
    #def __init__(self, filter_data, method="Old"):
    def __init__(self, data_file, filters=None, method="Old"):
        """
        Initialize the extinction correction class.

        :param data_file: Path to the data file containing the UVOT data.
        :param filters: List of filter names in the same order as the data file.
                        Defaults to ["B", "U", "V", "M2", "W1", "W2"].
        :param method: "Old" to use ExtinctionOld or "New" to use ExtinctionNew methods.
        """
        if filters is None:
            filters = ["B", "U", "V", "M2", "W1", "W2"]
        
        data = np.loadtxt(data_file)
        #self.data = data_file
        
        # Convert the data to a dictionary
        #self.filter_data = {filters[i]: list(self.data[i]) for i in range(len(filters))}
        self.filter_data = {filters[i]: list(data[i]) for i in range(len(filters))}
 
        #self.filter_data = filter_data
        self.method = method
        self.wavelengths = {
            "B": 4392,
            "U": 3465,
            "V": 5468,
            "M2": 2246,
            "W1": 2600,
            "W2": 1928,
        }
        self.frequencies = {
            "B": 691105467312000,
            "U": 876452462418000,
            "V": 550993265238000,
            "M2": 1351400822148000,
            "W1": 1173903970434000,
            "W2": 1601439013434000,
        }
        

    def perform_correction(self):
        """
        Perform extinction correction for all available filters.
        
        return: List of results containing corrected flux data for each filter.
        """
        results = []
        methods = {
            "Old": {
                "B": ExtinctionOld.extinction_B,
                "U": ExtinctionOld.extinction_U,
                "V": ExtinctionOld.extinction_V,
                "M2": ExtinctionOld.extinction_M2,
                "W1": ExtinctionOld.extinction_W1,
                "W2": ExtinctionOld.extinction_W2,
            },
            "New": {
                "B": ExtinctionNew.extinction_B1,
                "U": ExtinctionNew.extinction_U1,
                "V": ExtinctionNew.extinction_V1,
                "M2": ExtinctionNew.extinction_M21,
                "W1": ExtinctionNew.extinction_W11,
                "W2": ExtinctionNew.extinction_W21,
            },
        }
        correction_methods = methods[self.method]

        for filter_name, data in self.filter_data.items():
            mag, dmag, ebv_A = data
            method = correction_methods[filter_name]
            result = (
                method(ebv_A, mag, dmag) if self.method == "Old" else method(mag, dmag, ebv_A)
            )
            results.append((filter_name, self.frequencies[filter_name], *result))

        return results

    def save_results(self, results, output_file="corrected_data"):
        """
        Save the corrected results to a file.
        return: a output file contains " frequencies,
                                         corrected magnitude,
                                         Counts,
                                         corrected flux_density,
                                         corrected flux_density error,
                                         corrected Flux,
                                         corrected Flux error
        """
        
        self.filename = f"{output_file}.txt"
        
        with open(self.filename, "w") as f:
            f.write("Filter Wavelength Corrected_Mag Count_Rate Flux_Density eFlux_Density Flux eFlux\n")
            for res in results:
                #f.write(f"{res[0]} {res[1]} {' '.join(map(str, res[2:]))}\n")
                f.write(f"{res[1]} {' '.join(map(str, res[2:]))}\n")
        print(f"Results saved to {output_file}")

    def plot_results(self, results):
        """
        Plot the corrected flux with error bars as a function of energy (SED).
        """
        
        frequencies = [res[1] for res in results]
        fluxes = [res[6] for res in results]
        errors = [res[7] for res in results]

        plt.figure(figsize=(10, 7),dpi=800)
        plt.errorbar(frequencies, fluxes, yerr=errors, fmt="o", color="k", ecolor="black", alpha=0.7,capsize=3)
        plt.ylabel(r'$\nu F_{\nu} \ \mathrm{ergs} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}$',
               fontsize=25)
        plt.xlabel(r'$\nu \ (\mathrm{Hz})$', fontsize=25)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.loglog()
        plt.title("Corrected UVOT spectra")
        plt.tight_layout()
        plt.show()


