from .SIConfigFile import SIConfigFile
import time
import numpy as np
import random
from datetime import datetime
from .SIException import SIException
from .Region import Region


class SimulatION(object):
    """The SimulatION class represents an experiment that can be run with the MinION DNA sequencer. The user has to specify some values, like genome length, coverage, a list of regions, and a configuration file."""

    def __init__(self):
        self._region_list = []
        self._cfile = None
        self._genome_length = 5000000
        self._coverage = 60

    @property
    def genome_length(self):
        """The genome_length property. Expected value: integer, minimum value of 1.
    Determines the length of the genome which is to be sequenced."""
        return self._genome_length

    @genome_length.setter
    def genome_length(self, value):
        try:
            value = int(value)
        except Exception:
            raise SIException("Please enter a valid number for genome length.")
        else:
            if value < 1:
                raise SIException("Genome length can not be less than one.")
            else:
                self._genome_length = value

    @genome_length.deleter
    def genome_length(self):
        del self._genome_length

    def coverage():
        doc = """The coverage property. Expected value: integer, minimum value of 1.
    Determines, how often a region has to be read by the MinION to be classified as covered."""

        def fget(self):
            return self._coverage

        def fset(self, value):
            try:
                value = int(value)
            except Exception:
                raise SIException("Please enter a valid number for coverage.")
            else:
                if value < 1:
                    raise SIException("Coverage can not be less than one.")
                else:
                    self._coverage = value

        def fdel(self):
            del self._coverage

        return locals()

    coverage = property(**coverage())

    def cfile():
        doc = """The cfile property. Expected value: a instance of the class MTSConfigFile."""

        def fget(self):
            return self._cfile

        def fset(self, value):
            if isinstance(value, SIConfigFile) or value == None:
                self._cfile = value
            else:
                raise SIException("Please specify a valid Config File for cfile.")

        def fdel(self):
            del self._cfile

        return locals()

    cfile = property(**cfile())

    def region_list():
        doc = """The region_list property. Expected value: a list of instances of the class Region or an empty list."""

        def fget(self):
            return self._region_list

        def fset(self, value):
            if isinstance(value, list):
                if len(value) == 0:
                    self._region_list = value
                else:
                    for item in value:
                        if not isinstance(item, Region):
                            raise SIException("Please enter valid regions.")
                    self._region_list = value
            else:
                raise SIException("Please enter valid regions.")

        def fdel(self):
            del self._region_list

        return locals()

    region_list = property(**region_list())

    def calculate_time(self):
        """Calculates the time the MinION will take to run an experiment with the given properties and configuration file.
    No parameters.
    Returns: expected period of time in hours, represented by a float."""
        if self.validate_parameters():
            self.coverage_list = np.zeros(shape=int(self.genome_length), dtype=int)
            active_pores = np.zeros(shape=int(self.cfile.max_active_pores), dtype=float)
            start_time = datetime.now()

            while True:
                r = random.random()
                length = 0
                for pair in self.cfile.read_length_distribution:
                    r -= pair[1]
                    if r <= 0:
                        length = pair[0]
                        # print "length:", length
                        break
                start = int(random.random() * (self.genome_length - length))
                end = start + length
                if len(self.cfile.wear_out) > 0:
                    sec = active_pores.min()
                    # print "sec:", sec
                    try:
                        # print int(sec)
                        pores = self.cfile.wear_out[int(sec)]
                        # pores = 1
                    except Exception as e:
                        # print e
                        # print e
                        pores = 1
                    finally:
                        if pores == 0:
                            pores = 1
                        if len(active_pores) > pores:
                            sorted_pores = np.sort(active_pores)
                            active_pores, x = np.split(sorted_pores, [pores])
                        elif len(active_pores) < pores:
                            new_pores_num = pores - len(active_pores)
                            new_pores = np.empty(new_pores_num)
                            new_pores.fill(sec)
                            active_pores = np.append(active_pores, new_pores)
                            # print "pores:", pores
                            # print "active pores:", len(active_pores)
                pore = active_pores.argmin()
                decision = True
                if not self.cfile.read_until == None:
                    decision, needed_time = self.cfile.read_until.decide(self.is_in_region(start, end),
                                                                         self.cfile.bases_per_second)
                    if not decision:
                        active_pores[pore] += needed_time
                if self.cfile.read_until == None or decision:
                    secs = float(length) / self.cfile.bases_per_second
                    # print "secs:", secs
                    active_pores[pore] += secs
                    self.coverage_list[start:end + 1] += 1
                if self.regions_are_covered():
                    break

            end_time = datetime.now()
            calc_time = (end_time - start_time).seconds
            time_in_secs = active_pores.max()
            time_in_hrs = time_in_secs / 3600
            return time_in_hrs

    def regions_are_covered(self):
        """Checks, if all regions within region_list are classified as covered.
    No parameters.
    Returns: True, if all regions are covered; False otherwise"""
        for region in self.region_list:
            if np.any(self.coverage_list[region.start:region.end + 1] < self.coverage):
                return False
        return True

    def is_in_region(self, start, end):
        """Checks, if a certain read is (partly) located within at least one region within region_list.
    Parameters:
    start: int; the location of the start of the read within the genome (The first base of the genome is numbered 0.)
    end: int; the location of the end of the read within the genome (The first base of the genome is numbered 0.)

    Returns: True, if the read is within at least one region; False otherwise"""
        for region in self.region_list:
            if (start >= region.start and start <= region.end) or (end >= region.start and end <= region.end):
                return True
        return False

    def add_region(self, name, start, end, chr_name):
        """Creates a new instance of the class Region, validates the properties, and adds it to region_list, if validation does not fail.
    Parameters:
    name: string; Name of the Region
    start: int; the location of the start of the region within the genome (The first base of the genome is numbered 0.)
    end: int; the location of the end of the region within the genome (The first base of the genome is numbered 0.)
    chr_name: string; the name of the chromosome within which the region is located.

    Region is not added to region_list if:
    - start is greater or equal to genome_length
    - end is greater or equal to genome_length
    - there is already a region in region_list with (partly) covers the new region
    - name is empty
    - or assigning one of the values fails
    Raises SIException in case of failure.

    Returns nothing."""
        try:
            region = Region()
            region.name = name
            region.start = start
            region.end = end
            region.chr_name = chr_name
        except Exception as e:
            raise SIException(e.message)
        else:
            if region.start >= self._genome_length:
                raise SIException("Start can not be greater than genome length.")
            elif region.end >= self._genome_length:
                raise SIException("End can not be greater than genome length.")
                # elif end < start:
                # raise SIException("End can not be less than start")
            # elif self.is_in_region(region.start, region.end):
            #  raise SIException("This region is already (partly) covered by another region.")
            elif name == "":
                raise SIException("Name can not be empty.")
            else:
                self.region_list.append(region)

    def region_bigger_genome_length(self):
        """Checks if any region within region_list is (partly) outside of the genome.
    No parameters.
    Returns: True, if a region is (partly) outside of the genome; False otherwise"""
        for region in self.region_list:
            if region.start >= self.genome_length or region.end >= self.genome_length:
                return True
        return False

    def validate_parameters(self):
        """Validates the combination of the properties. Also calls MTSConfigFile.validate(), if cfile is specified.
    No parameters.
    Raises SIException if:
    - No configuration file is specified (cfile is None)
    - region_bigger_genome_length returns True
    - No regions are specified

    Returns True if no exception is raised.
    """
        if self.cfile == None:
            raise SIException("Please specify a configuration file.")
        else:
            if self.cfile.validate(self.genome_length):
                if self.region_bigger_genome_length():
                    raise SIException("All regions must lie within genome.")
                elif len(self.region_list) == 0:
                    raise SIException("Please specify regions.")
                else:
                    return True
            else:
                return False

    def regions_from_bedfile(self, bedfile):
        """Opens and reads a BED-file, identifies regions within the file and adds them to region_list using add_region().
    Parameters:
    bedfile: path to the BED-file

    Returns nothing."""
        with open(bedfile, "rb") as bed_file:
            for line in bed_file:
                info = line.split()
                if not info[0] == "track" and not info[0] == "browser" and not info[0] == "#":
                    if len(info) > 3:
                        name = info[3]
                    else:
                        name = info[0]
                    end = int(info[2]) - 1
                    self.add_region(name, info[1], end, info[0])

    def regions_to_bedfile(self, bedfile):
        """Opens a BED-file and writes the region_list into the file in appropriate style.
    Parameters:
    bedfile: path to the BED-file

    Returns nothing."""
        with open(bedfile, "wb+") as bed_file:
            bed_file.write('''track name="BedFileBySI" description="Bed file generated by SimulatION"''')
            for region in self.region_list:
                bed_file.write(
                    "\n" + region.chr_name + " " + str(region.start) + " " + str(region.end + 1) + " " + region.name)
            bed_file.flush()

    def regions_to_bedfile_web(self, bed_instance):
        """Works the same way as regions_to_bedfile(), but instead of writing to a BED-file, it writes to an instance of any class, which implements the write()-method.
    Parameters:
    bed_instance: an instance of any class which implements the write()-method, for example string or a Django-HttpResponse

    Returns nothing.."""
        bed_instance.write('''track name="BedFileBySI" description="Bed file generated by SimulatION"''')
        for region in self.region_list:
            bed_instance.write(
                "\n" + region.chr_name + " " + str(region.start) + " " + str(region.end + 1) + " " + region.name)
