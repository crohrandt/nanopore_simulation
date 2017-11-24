from .SIException import SIException
import random


class ReadUntil(object):
    def __init__(self):
        self._bases_number = 20
        self._decision_time = 10
        self._error_rate_fp = 0.2
        self._error_rate_fn = 0.3

    def bases_number():
        doc = """The bases_number property. Describes, how many bases are needed to be read, until a decision can be made.
    Expected value: integer"""

        def fget(self):
            return self._bases_number

        def fset(self, value):
            try:
                value = int(value)
            except Exception as e:
                raise SIException("Please enter a valid value for bases number: " + e.message)
            else:
                self._bases_number = value

        def fdel(self):
            del self._bases_number

        return locals()

    bases_number = property(**bases_number())

    def decision_time():
        doc = """The decision_time property. Describes, how many seconds it takes to make a decision.
    Expected value: integer"""

        def fget(self):
            return self._decision_time

        def fset(self, value):
            try:
                value = float(value)
            except Exception as e:
                raise SIException("Please enter a valid value for decision time: " + e.message)
            else:
                self._decision_time = value

        def fdel(self):
            del self._decision_time

        return locals()

    decision_time = property(**decision_time())

    def error_rate_fp():
        doc = """The error_rate_fp property. Describes, how likely it is, that a read is falsely considered to be in a region that is to be covered.
    Expected value: float between 0 and 1."""

        def fget(self):
            return self._error_rate_fp

        def fset(self, value):
            try:
                value = float(value)
            except Exception:
                raise SIException("Please enter a valid value for the error rate (false positive): " + e.message)
            else:
                self._error_rate_fp = value

        def fdel(self):
            del self._error_rate_fp

        return locals()

    error_rate_fp = property(**error_rate_fp())

    def error_rate_fn():
        doc = """The error_rate_fn property. Describes, how likely it is, that a read is falsely considered not to be in a region that is to be covered.
      Expected value: float between 0 and 1."""

        def fget(self):
            return self._error_rate_fn

        def fset(self, value):
            try:
                value = float(value)
            except Exception as e:
                raise SIException("Please enter a valid number for the error rate (false negative): " + e.message)
            else:
                self._error_rate_fn = value

        def fdel(self):
            del self._error_rate_fn

        return locals()

    error_rate_fn = property(**error_rate_fn())

    def decide(self, is_in_region, bases_per_second):
        """Decides, according to the error rates, whether the given read is in any of the regions and calculates the time needed to make that decision.
    Parameters:
    is_in_region: boolean; determines, whether the read is actually in any region (output of SimulatION.is_in_region)
    bases_per_second: int; determines, how many bases are read per second by the pore (value of SIConfigFile.bases_per_second)
    Returns:
    decision: boolean; True, if the read is supposed to be in any region, False otherwise
    needed_time: float; the time needed to make that decision"""
        needed_time = (float(self.bases_number) / bases_per_second) * 2 + self.decision_time
        rnd = random.random()
        if is_in_region:
            if rnd < self.error_rate_fn:
                return False, needed_time
            else:
                return True, needed_time
        else:
            if rnd < self.error_rate_fp:
                return True, needed_time
            else:
                return False, needed_time
