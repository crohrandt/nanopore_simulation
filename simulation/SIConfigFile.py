import pickle
import pandas as pd
import math
import numpy as np
import os
import h5py
from .SIException import SIException
from .ReadUntil import ReadUntil


class SIConfigFile(object):
    """
  This class can open and store configuration files for the MinION Time Simulator.
  """

    def __init__(self):
        self._read_length_distribution = []
        self._event_length_distribution = []
        self._bases_per_second = 450
        self._pores_number = 1500
        self._max_active_pores = 512
        self._read_until = None
        self._wear_out = []
        self._file_path = None
        self._ranges = []
        self._offsets = []
        self._asic_temps = []
        self._heatsink_temps = []
        self._sampling_rate = 4000
        self._protocols_version = None
        self._version_number = None
        self._sample_id = "Sample_ID"
        self._digitisation = 8192.0
        self._experiment_kit = None
        self._asic_id = None
        self._asic_id_eeprom = None
        self._auto_update = None
        self._auto_update_source = None
        self._bream_core_version = None
        self._bream_is_standard = None
        self._bream_nc_version = None
        self._bream_ont_version = None
        self._bream_prod_version = None
        self._bream_rnd_version = None
        self._exp_script_name = None
        self._installation_type = None
        self._local_firmware_file = None
        self._operating_system = None
        self._protocol_run_id = None
        self._protocols_version = None
        self._run_id = None
        self._usb_config = None
    
    def usb_config():
        doc = """The usb_config property.
    Expected value: str; minimum value of 1."""

        def fget(self):
            return self._usb_config

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid number for bases per second.")
            else:
                self._usb_config = value

        def fdel(self):
            del self._usb_config

        return locals()

    usb_config = property(**usb_config())
    
    def run_id():
        doc = """The run_id property.
    Expected value: str."""

        def fget(self):
            return self._run_id

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid run id.")
            else:
                self._run_id = value

        def fdel(self):
            del self._run_id

        return locals()

    run_id = property(**run_id())
    
    def protocols_version():
        doc = """The protocols_version property.
    Expected value: str."""

        def fget(self):
            return self._protocols_version

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid protocols version.")
            else:
                self._protocols_version = value

        def fdel(self):
            del self._protocols_version

        return locals()

    protocols_version = property(**protocols_version())

    def version_number():
        doc = """The version_number property.
       Expected value: str."""

        def fget(self):
            return self._version_number

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid protocols version.")
            else:
                self._version_number = value

        def fdel(self):
            del self._version_number

        return locals()

    version_number = property(**version_number())
    
    def protocol_run_id():
        doc = """The protocol_run_id property.
    Expected value: str."""

        def fget(self):
            return self._protocol_run_id

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid protocol run id.")
            else:
                self._protocol_run_id = value

        def fdel(self):
            del self._protocol_run_id

        return locals()

    protocol_run_id = property(**protocol_run_id())
    
    def operating_system():
        doc = """The operating_system property.
    Expected value: str."""

        def fget(self):
            return self._operating_system

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid operating system.")
            else:
                self._operating_system = value

        def fdel(self):
            del self._operating_system

        return locals()

    operating_system = property(**operating_system())
    
    def local_firmware_file():
        doc = """The local_firmware_file property.
    Expected value: str."""

        def fget(self):
            return self._local_firmware_file

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid local firmware file.")
            else:
                self._local_firmware_file = value

        def fdel(self):
            del self._local_firmware_file

        return locals()

    local_firmware_file = property(**local_firmware_file())
    
    def installation_type():
        doc = """The installation_type property.
    Expected value: str."""

        def fget(self):
            return self._installation_type

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid instalation type.")
            else:
                self._installation_type = value

        def fdel(self):
            del self._installation_type

        return locals()

    installation_type = property(**installation_type())

    def heatsink_temps():
        doc = """The heatsink_temps property.
       Expected value: empty list.
       To add probabilities, use add_ld(self.read_length_ditribution)."""

        def fget(self):
            return self._heatsink_temps

        def fset(self, value):
            if value == []:
                self._heatsink_temps = value
            else:
                raise SIException(
                    "You can only assign empty lists to read length distribution. Please use add_ld(self.heatsink_temps) to add probabilities.")

        def fdel(self):
            del self._heatsink_temps

        return locals()

    heatsink_temps = property(**heatsink_temps())
    
    def exp_script_name():
        doc = """The exp_script_name property.
    Expected value: str."""

        def fget(self):
            return self._exp_script_name

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid experiment script name.")
            else:
                self._exp_script_name = value

        def fdel(self):
            del self._exp_script_name

        return locals()

    exp_script_name = property(**exp_script_name())
    
    def bream_rnd_version():
        doc = """The bream_rnd_version property.
    Expected value: str."""

        def fget(self):
            return self._bream_rnd_version

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid bream rnd version.")
            else:
                self._bream_rnd_version = value

        def fdel(self):
            del self._bream_rnd_version

        return locals()

    bream_rnd_version = property(**bream_rnd_version())
    
    def bream_prod_version():
        doc = """The bream_prod_version property.
    Expected value: str."""

        def fget(self):
            return self._bream_prod_version

        def fset(self, value):
            try:
                value = str(value)
            except:
                raise SIException("Please enter a valid bream prod version.")
            else:
                self._bream_prod_version = value

        def fdel(self):
            del self._bream_prod_version

        return locals()

    bream_prod_version = property(**bream_prod_version())
    
    def bream_ont_version():
        doc = """The bream_ont_version property.
    Expected value: str."""

        def fget(self):
            return self._bream_ont_version

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid bream ont version.")
            else:
                self._bream_ont_version = value

        def fdel(self):
            del self._bream_ont_version

        return locals()

    bream_ont_version = property(**bream_ont_version())
    
    def bream_nc_version():
        doc = """The bream_nc_version property.
    Expected value: str."""

        def fget(self):
            return self._bream_nc_version

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid bream nc version.")
            else:
                self._bream_nc_version = value

        def fdel(self):
            del self._bream_nc_version

        return locals()

    bream_nc_version = property(**bream_nc_version())
    
    def bream_is_standard():
        doc = """The bream_is_standard property.
    Expected value: str."""

        def fget(self):
            return self._bream_is_standard

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid standard for bream.")
            else:
                self._bream_is_standard = value

        def fdel(self):
            del self._bream_is_standard

        return locals()

    bream_is_standard = property(**bream_is_standard())
    
    def bream_core_version():
        doc = """The bream_core_version property.
    Expected value: str."""

        def fget(self):
            return self._bream_core_version

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid bream core version.")
            else:
                self._bream_core_version = value

        def fdel(self):
            del self._bream_core_version

        return locals()

    bream_core_version = property(**bream_core_version())
    
    def auto_update_source():
        doc = """The auto_update_source property.
    Expected value: str."""

        def fget(self):
            return self._auto_update_source

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid number for bases per second.")
            else:
                self._auto_update_source = value

        def fdel(self):
            del self._auto_update_source

        return locals()

    auto_update_source = property(**auto_update_source())
    
    def auto_update():
        doc = """The auto_update property.
    Expected value: str."""

        def fget(self):
            return self._auto_update

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid value for auto update.")
            else:
                self._auto_update = value

        def fdel(self):
            del self._auto_update

        return locals()

    auto_update = property(**auto_update())

    def asic_temps():
        doc = """The asic_temps property.
       Expected value: empty list.
       To add probabilities, use add_ld(self.read_length_ditribution)."""

        def fget(self):
            return self._asic_temps

        def fset(self, value):
            if value == []:
                self._asic_temps = value
            else:
                raise SIException(
                    "You can only assign empty lists to read length distribution. Please use add_ld(self.asic_temps) to add probabilities.")

        def fdel(self):
            del self._asic_temps

        return locals()

    asic_temps = property(**asic_temps())
    
    def asic_id_eeprom():
        doc = """The asic_id_eeprom property.
    Expected value: str."""

        def fget(self):
            return self._asic_id_eeprom

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid asic eeprom id.")
            else:
                self._asic_id_eeprom = value

        def fdel(self):
            del self._asic_id_eeprom

        return locals()

    asic_id_eeprom = property(**asic_id_eeprom())
    
    def asic_id():
        doc = """The asic_id property.
    Expected value: str."""

        def fget(self):
            return self._asic_id

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid asic id.")
            else:
                self._asic_id = value

        def fdel(self):
            del self._asic_id

        return locals()

    asic_id = property(**asic_id())
    
    def experiment_kit():
        doc = """The experiment_kit property.
    Expected value: str."""

        def fget(self):
            return self._experiment_kit

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid experiment kit.")
            else:
                self._experiment_kit = value

        def fdel(self):
            del self._experiment_kit

        return locals()

    experiment_kit = property(**experiment_kit())
    
    def digitisation():
        doc = """The digitasation property.
    Expected value: int; minimum value of 1."""

        def fget(self):
            return self._digitisation

        def fset(self, value):
            try:
                value = int(value)
            except:
                raise SIException("Please enter a valid digitisation.")
            else:
                if value < 1:
                    raise SIException("Digitisation can not be less than one.")
                else:
                    self._digitisation = value

        def fdel(self):
            del self._digitisation

        return locals()

    digitisation = property(**digitisation())
    
    def sample_id():
        doc = """The sample_id property.
    Expected value: str."""

        def fget(self):
            return self._sample_id

        def fset(self, value):
            try:
                value = str(value)
            
            except:
                raise SIException("Please enter a valid sample id.")
            else:
                self._sample_id = value

        def fdel(self):
            del self._sample_id

        return locals()

    sample_id = property(**sample_id())

    def sampling_rate():
        doc = """The sampling_rate property.
    Expected value: int; minimum value of 1."""

        def fget(self):
            return self._sampling_rate

        def fset(self, value):
            try:
                value = int(value)
            except:
                raise SIException("Please enter a valid number for the sampling rate.")
            else:
                if value < 1:
                    raise SIException("Samplingrate can not be less than one.")
                else:
                    self._sampling_rate = value

        def fdel(self):
            del self._sampling_rate

        return locals()

    sampling_rate = property(**sampling_rate())
    
    def bases_per_second():
        doc = """The bases_per_second property.
    Expected value: int; minimum value of 1."""

        def fget(self):
            return self._bases_per_second

        def fset(self, value):
            try:
                value = int(value)
            except:
                raise SIException("Please enter a valid number for bases per second.")
            else:
                if value < 1:
                    raise SIException("Bases per second can not be less than one.")
                else:
                    self._bases_per_second = value

        def fdel(self):
            del self._bases_per_second

        return locals()

    bases_per_second = property(**bases_per_second())

    def pores_number():
        doc = """The pores_number property.
    Expected value: int; minimum value of 1."""

        def fget(self):
            return self._pores_number

        def fset(self, value):
            try:
                value = int(value)
            except:
                raise SIException("Please enter a valid number for pores.")
            else:
                if value < 1:
                    raise SIException("Pores number can not be less than one.")
                else:
                    self._pores_number = value

        def fdel(self):
            del self._pores_number

        return locals()

    pores_number = property(**pores_number())

    def max_active_pores():
        doc = """The max_active_pores property.
    Expected value: int; minimum value of 1. Can not be greater than pores_number."""

        def fget(self):
            return self._max_active_pores

        def fset(self, value):
            try:
                value = int(value)
            except:
                raise SIException("Please enter a valid number for max. active pores.")
            else:
                if value < 1:
                    raise SIException("Max. active pores can not be less than one.")
                elif value > self.pores_number:
                    raise SIException("Max. active pores can not be greater than pores number.")
                else:
                    self._max_active_pores = value

        def fdel(self):
            del self._max_active_pores

        return locals()

    max_active_pores = property(**max_active_pores())

    def read_until():
        doc = """The read_until property. Holds an instance of ReadUntil.
    Expected value: instance of ReadUntil or None."""

        def fget(self):
            return self._read_until

        def fset(self, value):
            if isinstance(value, ReadUntil) or value == None:
                self._read_until = value
            else:
                raise SIException("Please enter a valid instance or None for read until.")

        def fdel(self):
            del self._read_until

        return locals()

    read_until = property(**read_until())

    def file_path():
        doc = """The file_path property.
    Expected value: string."""

        def fget(self):
            return self._file_path

        def fset(self, value):
            self._file_path = value

        def fdel(self):
            del self._file_path

        return locals()

    file_path = property(**file_path())

    def read_length_distribution():
        doc = """The read_length_distribution property.
    Expected value: empty list.
    To add probabilities, use add_ld(self.read_length_ditribution)."""

        def fget(self):
            return self._read_length_distribution

        def fset(self, value):
            if value == []:
                self._read_length_distribution = value
            else:
                raise SIException(
                    "You can only assign empty lists to read length distribution. Please use add_ld(self.read_length_distribution) to add probabilities.")

        def fdel(self):
            del self._read_length_distribution

        return locals()

    read_length_distribution = property(**read_length_distribution())

    def wear_out():
        doc = """The wear_out property. Describes, how many pores are active in every second. Determined by a given CSV-file. If not given, the time is calculated assuming there are 512 pores active in every second.
    Expected value: numpy-array."""

        def fget(self):
            return self._wear_out

        def fset(self, value):
            if isinstance(value, np.ndarray) or value == []:
                self._wear_out = value
            else:
                raise SIException("Please enter a valid value for wear out.")

        def fdel(self):
            del self._wear_out

        return locals()

    wear_out = property(**wear_out())

    def offsets():
        doc = """The offsets is a distribution of all occuring offset values within the sample pool.
        Expected value: list of integers."""

        def fget(self):
            return self._offsets

        def fset(self, value):
            if isinstance(value, list) or value == []:
                self._offsets = value
            else:
                raise SIException("Please enter a valid value for offsets.")

        def fdel(self):
            del self._offsets

        return locals()

    offsets = property(**offsets())

    def ranges():
        doc = """The ranges is a distribution of all occuring range values within the sample pool.
        Expected value: list of integers."""

        def fget(self):
            return self._ranges

        def fset(self, value):
            if isinstance(value, list) or value == []:
                self._ranges = value
            else:
                raise SIException("Please enter a valid value for ranges.")

        def fdel(self):
            del self._ranges

        return locals()

    ranges = property(**ranges())

    def event_length_distribution():
        doc = """The event_length_distribution property.
            Expected value: empty list.
            To add probabilities, use add_ld(self.event_length_ditribution)."""

        def fget(self):
            return self._event_length_distribution

        def fset(self, value):
            if value == []:
                self._event_length_distribution = value
            else:
                raise SIException(
                    "You can only assign empty lists to event length distribution. Please use add_ld(self.event_length_distribution) to add probabilities.")

        def fdel(self):
            del self._event_length_distribution

        return locals()

    event_length_distribution = property(**event_length_distribution())

    def load_file(self, pklfile):
        """Loads the contents of an opened file using pickle and assignes the properties.
    Parameters:
    pklfile: an opened file

    Returns True, if everything works."""
        contents = pickle.load(pklfile)
        self.read_length_distribution = []
        for rld in contents[0]:
            self.add_distribution(self.read_length_distribution, rld[0], rld[1])
        self.bases_per_second = contents[1]
        self.pores_number = contents[2]
        self.max_active_pores = contents[3]
        self.read_until = contents[4]
        self.wear_out = contents[5]
        for eld in contents[6]:
            self.add_distribution(self.event_length_distribution, eld[0], eld[1])
        self.offsets = contents[7]
        self.ranges = contents[8]
        self.sampling_rate = contents[9]
        self.protocols_version = contents[10]
        self.version_number = contents[11]
        self.sample_id = contents[12]
        self.digitisation = contents[13]
        self.experiment_kit = contents[14]
        self.asic_id = contents[15]
        self.asic_id_eeprom = contents[16]
        self.asic_temp = contents[17]
        self.auto_update = contents[18]
        self.auto_update_source = contents[19]
        self.bream_core_version = contents[20]
        self.bream_is_standard = contents[21]
        self.bream_nc_version = contents[22]
        self.bream_ont_version = contents[23]
        self.bream_prod_version = contents[24]
        self.bream_rnd_version = contents[25]
        self.exp_script_name = contents[26]
        self.heatsink_temp = contents[27]
        self.installation_type = contents[28]
        self.local_firmware_file = contents[29]
        self.operating_system = contents[30]
        self.protocol_run_id = contents[31]
        self.protocols_version = contents[32]
        self.run_id = contents[33]
        self.usb_config = contents[34]

        return True

    def save_file(self, pklfile):
        """Writes the properties to an opened file.
    Parameters:
    pklfile: an opened file

    Returns True, if everything works."""
        '''contents = [self._read_length_distribution, self._bases_per_second, self._pores_number, self._max_active_pores, self._read_until, self._wear_out]
    pickle.dump(contents, pklfile, protocol=2)'''
        self.save_file_web(pklfile)
        return True

    def save_file_web(self, instance):
        """Works the same way as save_file(), but instead of writing to a file, it writes to an instance of any class which implements the write()-method.
    Parameters:
    instance: an instance of any class which implements the write()-method, for example string or a Django HttpResponse.

    Returns nothing."""
        contents = [self._read_length_distribution, self._bases_per_second, self._pores_number, self._max_active_pores,
                    self._read_until, self._wear_out, self.event_length_distribution, self.offsets, self.ranges,
                    self.sampling_rate, self.protocols_version, self.version_number, self.sample_id, self.digitisation, 
                    self.experiment_kit, self.asic_id, self.asic_id_eeprom, self.asic_temps, self.auto_update,
                    self.auto_update_source, self.bream_core_version, self.bream_is_standard, self.bream_nc_version,
                    self.bream_ont_version, self.bream_prod_version, self.bream_rnd_version, self.exp_script_name,
                    self.heatsink_temps, self.installation_type, self.local_firmware_file, self.operating_system,
                    self.protocol_run_id, self.protocols_version, self.run_id, self.usb_config]
        instance.write(pickle.dumps(contents, protocol=2))

    def addup(self,distribution):
        """Adds up the probabilities of read_length_distribution.
    No parameters.

    Returns the sum of the probabilities."""
        return sum(x[1] for x in distribution)

    def length_bigger_genome_length(self, genome_length):
        """Checks, if any read length within read_length_distribution is greater than the given genome_length.
    Parameters:
    genome_length: int

    Returns: True, if there is a read length which is greater than the genome_length; False otherwise."""
        for rld in self.read_length_distribution:
            if rld[0] >= genome_length:
                return True
        return False

    def get_distribution_probabilities(self, distribution):
        return list(zip(*distribution))[1]

    def get_distribution_keys(self, distribution):
        return list(zip(*distribution))[0]

    def add_distribution(self, distribution, length, probability):
        """Validates the parameters and adds a tupel to read_length_distribution, if validation does not fail.
    Parameters:
    length: int; the read length
    probability: float; the probability, with which the read length occurs

    Validation fails if:
    - length can not be casted to int
    - probability can not be casted to float
    - probability is greater than 1
    - addup() + probability is bigger than 1
    Raises SIException on failure.

    Returns nothing."""
        try:
            length = length
            probability = float(probability)
        except ValueError:
            raise SIException("Please enter valid numbers (int for length, float for probability)!")
        else:
            if probability > 1:
                raise SIException("The probability can not be greater than 1.")
            elif self.addup(distribution) + probability > (1 + 1e-09):
                raise SIException("All probabilities must add up to exactly 1.")
            else:
                distribution.append((length, probability))

    def distribution_addup_is_valid(self):
        """Checks, whether the sum of the probabilities in read_length_distribution is (close to) 1.
    No parameters.
    Returns True, if sum is close to 1; Else otherwise.
    """
        return isclose(self.addup(self.read_length_distribution), 1.0) or isclose(self.addup(self.event_length_distribution), 1.0)

    def validate(self, genome_length):
        """Validates the properties.
    Parameters:
    genome_length: int

    Raises SIException if:
    - length_bigger_genome_length() returns True
    - rld_addup_is_valid() returns False

    Returns True, if validation succeeds."""
        if self.length_bigger_genome_length(genome_length):
            raise SIException("Read length must be less than genome length.")
        elif not self.distribution_addup_is_valid():
            raise SIException("All probabilities must add up to exactly 1.")
        else:
            return True

    def parameters_from_csv(self, csv_file, read_length_name=None, start_time_name=None, duration_name=None):
        """If read_length_name is given:
    Calculates a read_length_distribution from a CSV-file, which contains a list of reads and their length, using pandas and replaces the current read_length_distribution.
    If start_time_name and duration_name are given:
    Calculates, how many pores are active in every second. Calculates the maximum of active pores.
    If read_length_name and duration_name are given:
    Calculates, how many bases per second are read on average.
    Parameters:
    csv_file: the path to the CSV-file
    read_length_name: string; the name of the column within the CSV-file which holds the read length
    start_time_name: string; the name of the column within the CSV-file which holds the start time of the read
    duration_name: string; the name of the column within the CSV-file which holds the duration of the read

    Raises SIException on failure.

    Returns nothing."""
        try:
            usecols = []
            if read_length_name:
                usecols.append(read_length_name)
            if start_time_name and duration_name:
                usecols.append(start_time_name)
                usecols.append(duration_name)
            data = pd.read_csv(csv_file, sep=None, engine="python", usecols=usecols)
            if read_length_name:
                max_read_length = data[read_length_name].max()
                maximum = max_read_length + 2000
                sequence = range(0, maximum, 1000)
                labels = range(1000, maximum, 1000)
                cuts = pd.cut(data[read_length_name], sequence, labels=labels, include_lowest=True)
                series = pd.Series(cuts)
                counts = series.value_counts(normalize=True)
                self._read_length_distribution = list(counts.iteritems())
            if start_time_name and duration_name:
                start = int(min(data[start_time_name]))
                wearout = np.zeros(shape=int(max(data[start_time_name] + data[duration_name])) - start, dtype=int)
                for i in range(0, len(wearout)):
                    pos = start + i
                    active_pores = data.query(start_time_name + " <= " + str(
                        pos) + " and " + start_time_name + "+" + duration_name + " >= " + str(pos))
                    wearout[i] = len(active_pores)
                self._wear_out = wearout
                self._max_active_pores = max(wearout)
                if self.max_active_pores > self.pores_number:
                    self.pores_number = self.max_active_pores
            if read_length_name and duration_name:
                means = data.mean()
                self.bases_per_second = means[read_length_name] / means[duration_name]
        except Exception as e:
            raise e

    def get_columnnames(self, filename):
        data = pd.read_csv(filename, sep=None, engine="python", skiprows=[1, 1000000000])
        return list(data)

    def parameters_from_fast5(self, path_to_files=None, basecall_group='000'):
        """Returns
        nothing.
        """
        p = {}
        offsets = {}
        ranges = {}
        asic_temps = {}
        heatsink_temps = {}
        num_events = 0
        num_reads = 0

        try:
            for root, dirs, files in os.walk(path_to_files):
                for file in files:
                    if file.endswith(".fast5"):
                        try:
                            hf = h5py.File(os.path.join(root, file), 'r')
                            Metadata_Source_file = os.path.join(root, file)
                            num_reads += 1
                            try:
                                read_id = hf["Raw"]["Reads"][list(hf["Raw"]["Reads"].keys())[0]].attrs["read_id"]
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find raw data in file and thus no corresponding read id.  Continuing with other files')
                                continue
                            try:
                                sampling_rate = hf["UniqueGlobalKey"]["channel_id"].attrs["sampling_rate"]
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find sampling rate attribute.  Continuing with other files')
                                continue
                            try:
                                offset = hf["UniqueGlobalKey"]["channel_id"].attrs["offset"]
                                if offset in offsets:
                                    offsets[offset] += 1.0
                                else:
                                    offsets[offset] = 1.0
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find offset attribute.  Continuing with other files')
                                continue
                            try:
                                range = hf["UniqueGlobalKey"]["channel_id"].attrs["range"]
                                if range in ranges:
                                    ranges[range] += 1.0
                                else:
                                    ranges[range] = 1.0
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find range attribute.  Continuing with other files')
                                continue

                            try:
                                asic_temp = float(hf["UniqueGlobalKey"]["tracking_id"].attrs["asic_temp"])
                                if asic_temp in asic_temps:
                                    asic_temps[asic_temp] += 1.0
                                else:
                                    asic_temps[asic_temp] = 1.0
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find asic temp attribute.  Continuing with other files')
                                continue

                            try:
                                heatsink_temp = float(hf["UniqueGlobalKey"]["tracking_id"].attrs["heatsink_temp"])
                                if heatsink_temp in heatsink_temps:
                                    heatsink_temps[heatsink_temp] += 1.0
                                else:
                                    heatsink_temps[heatsink_temp] = 1.0
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find heatsink temp attribute.  Continuing with other files')
                                continue

                            try:
                                table = hf['Analyses']['Basecall_1D_' + str(basecall_group)]['BaseCalled_template']['Events'].value
                                move_indexes = np.where(table["move"] >= 1)[0].tolist()
                                unique_events = np.array(np.split(table, move_indexes[1:]))

                                for event in unique_events[10:]: # leave out the long adapter-like events within basecalled reads
                                    num_events += 1
                                    event_length = 0
                                    for subevent in event:
                                        if subevent[3].dtype == 'float32':
                                            event_length += int(subevent[3] * sampling_rate)
                                        else:
                                            event_length += subevent[3]
                                    if event_length in p:
                                        p[event_length] += 1.0
                                    else:
                                        p[event_length] = 1.0
                            except KeyError:
                                print("Error: " + str(file) + ' Could not find event data in file.  Continuing with other files')
                                continue
                            except OSError:
                                print("Error: " + str(file) + ' could not open fast5-file. Maybe it\'s damaged.  Continuing eith other files')

                            hf.close()

                        except OSError:
                            print("Error: Opening " + str(file) + ' was not possible. Maybe it\'s damaged.  Continuing with other files')
                            continue

                        except Exception as e:
                            raise e
            try:
                hf = h5py.File(Metadata_Source_file, 'r')
            except OSError:
                print("Error: Opening " + str(file) + ' was not possible. Maybe it\'s damaged.  Continuing with other files')

            try:
                self.sampling_rate = hf["UniqueGlobalKey"]["channel_id"].attrs["sampling_rate"]
            except KeyError as e:
                self.sampling_rate = 4000
            try:
                self.protocols_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["protocols_version"], 'utf-8')
            except KeyError as e:
                self.protocols_version = None
            try:
                self.version_number = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["version"], 'utf-8')
            except KeyError as e:
                self.version_number = None
            try:
                self.sample_id = str(hf["UniqueGlobalKey"]["context_tags"].attrs["user_filename_input"], 'utf-8')
            except KeyError as e:
                self.sample_id = "Sample_ID"
            try:
                self.digitisation = hf["UniqueGlobalKey"]["channel_id"].attrs["digitisation"]
            except KeyError as e:
                self.digitisation = 8192
            try:
                self.experiment_kit = str(hf["UniqueGlobalKey"]["context_tags"].attrs["experiment_kit"], 'utf-8')
            except KeyError as e:
                self.experiment_kit = None
            try:
                self.asic_id = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["asic_id"], 'utf-8')
            except KeyError as e:
                self.asic_id = 0
            try:
                self.asic_id_eeprom = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["asic_id_eeprom"], 'utf-8')
            except KeyError as e:
                self.asic_id_eeprom = 0
            try:
                self.auto_update = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["auto_update"], 'utf-8')
            except KeyError as e:
                self.auto_update = None
            try:
                self.auto_update_source = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["auto_update_source"], 'utf-8')
            except KeyError as e:
                self.auto_update_source = None
            try:
                self.bream_core_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["bream_core_version"], 'utf-8')
            except KeyError as e:
                self.bream_core_version = None
            try:
                self.bream_nc_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["bream_nc_version"], 'utf-8')
            except KeyError as e:
                self.bream_nc_version = None
            try:
                self.bream_ont_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["bream_ont_version"], 'utf-8')
            except KeyError as e:
                self.bream_ont_version = None
            try:
                self.bream_prod_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["bream_prod_version"], 'utf-8')
            except KeyError as e:
                self.bream_prod_version = None
            try:
                self.bream_rnd_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["bream_rnd_version"], 'utf-8')
            except KeyError as e:
                self.bream_rnd_version = None
            try:
                self.exp_script_name = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["exp_script_name"], 'utf-8')
            except KeyError as e:
                self.exp_script_name = None
            try:
                self.installation_type = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["installation_type"], 'utf-8')
            except KeyError as e:
                self.installation_type = None
            try:
                self.local_firmware_file = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["local_firmware_file"], 'utf-8')
            except KeyError as e:
                self.local_firmware_file = None
            try:
                self.operating_system = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["operating_system"], 'utf-8')
            except KeyError as e:
                self.operating_system = None
            try:
                self.protocol_run_id = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["protocol_run_id"], 'utf-8')
            except KeyError as e:
                self.protocol_run_id = None
            try:
                self.protocols_version = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["protocols_version"], 'utf-8')
            except KeyError as e:
                self.protocols_version = None
            try:
                self.run_id = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["run_id"], 'utf-8')
            except KeyError as e:
                self.run_id = None
            try:
                self.usb_config = str(hf["UniqueGlobalKey"]["tracking_id"].attrs["usb_config"], 'utf-8')
            except KeyError as e:
                self.usb_config = None

            hf.close()

            for i in p:
                self.add_distribution(self.event_length_distribution, i, (p[i] / num_events))

            for i in offsets:
                self.add_distribution(self.offsets, i, (offsets[i] / num_reads))

            for i in ranges:
                self.add_distribution(self.ranges, i, (ranges[i] / num_reads))

            for i in asic_temps:
                self.add_distribution(self.asic_temps, i, (asic_temps[i] / num_reads))

            for i in heatsink_temps:
                self.add_distribution(self.heatsink_temps, i, (heatsink_temps[i] / num_reads))

        except Exception as e:
            raise e

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    '''
  Python 2 implementation of Python 3.5 math.isclose()
  https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993
  '''
    # sanity check on the inputs
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")

    # short circuit exact equality -- needed to catch two infinities of
    # the same sign. And perhaps speeds things up a bit sometimes.
    if a == b:
        return True

    # This catches the case of two infinities of opposite sign, or
    # one infinity and one finite number. Two infinities of opposite
    # sign would otherwise have an infinite relative tolerance.
    # Two infinities of the same sign are caught by the equality check
    # above.
    if math.isinf(a) or math.isinf(b):
        return False

    # now do the regular computation
    # this is essentially the "weak" test from the Boost library
    diff = math.fabs(b - a)
    result = (((diff <= math.fabs(rel_tol * b)) or
               (diff <= math.fabs(rel_tol * a))) or
              (diff <= abs_tol))
    return result
