from .SIException import SIException


class Region(object):
    def name():
        doc = """The name property.
    Expected value: string"""

        def fget(self):
            return self._name

        def fset(self, value):
            if value == "":
                raise SIException("Please enter a name.")
            else:
                self._name = value

        def fdel(self):
            del self._name

        return locals()

    name = property(**name())

    def start():
        doc = """The start property.
    Expected value: int; minimum value of 0. Represents the location of the first base within the region.
    The first base of the genome is numbered as 0."""

        def fget(self):
            return self._start

        def fset(self, value):
            try:
                value = int(value)
            except Exception:
                raise SIException("Please enter a valid number for region start.")
            else:
                if value < 0:
                    raise SIException("Region start can not be less than zero.")
                else:
                    self._start = value

        def fdel(self):
            del self._start

        return locals()

    start = property(**start())

    def end():
        doc = """The end property. Represents the location of the last base within the region. (CAUTION: in BED-files, end represents the location of the first base OUT of the region. This is handled by MinIONTimeSimulator.regions_from_bedfile().)
    Expected value: int; minimum value of start.
    The first base of the genome is numbered as 0."""

        def fget(self):
            return self._end

        def fset(self, value):
            try:
                value = int(value)
            except Exception as e:
                raise SIException("Please enter a valid number for region end. " + e.message)
            else:
                if value < self.start:
                    raise SIException("In " + self.name + ": Region end can not be less than region start.")
            self._end = value

        def fdel(self):
            del self._end

        return locals()

    end = property(**end())

    def chr_name():
        doc = """The chr_name property. Represents the name of the chromosome, within which the region lies.
    Expected value: string"""

        def fget(self):
            return self._chr_name

        def fset(self, value):
            if value == "":
                raise SIException("Please enter a chromesome name.")
            else:
                self._chr_name = value

        def fdel(self):
            del self._chr_name

        return locals()

    chr_name = property(**chr_name())
