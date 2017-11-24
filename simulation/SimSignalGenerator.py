from .SIConfigFile import SIConfigFile
from random import *
from Bio import SeqIO
from Bio import Seq
import numpy as np
from pylab import *
import scipy.signal

class SimSignalGenerator(object):

    def __init__(self,
                 config_file='',
                 debug=False):
        self.new_record = ""
        self.debug = debug
        self.ref_data = ""
        self.ref_data_keys = []
        self.ref_length = 0
        self.base = []
        self.k = 0
        self.case = 0
        self.min_length = 0
        self.max_length = 0
        self.signal_config = None
        self.read_length_distribution = []
        self.event_length_distribution = []
        self.offsets = []
        self.ranges = []
        self.bases_per_second = 0
        self.pores_number = 0
        self.max_active_pores = 0
        self.read_until = None
        self.wear_out = []
        self.file_path = None
        self.empty_regions = False
        self.model_data = None
        self.model_dict = {}
        self.load_config(config_file)

    def load_model(self, model_file):
        print("Loading model file")
        self.model_data = np.genfromtxt(model_file, delimiter="\t", dtype=None, comments='#', names=True)
        # k = kmer length
        self.k = len(self.model_data[0][0])
        self.model_dict = dict([(x[0].decode('utf-8'), (x[1], x[2])) for x in self.model_data])

    def load_reference(self, ref_file, low_mem=False, filter="None"):
        print("Loading reference genome. May take some time depending on genome size.")
        if low_mem:
            self.ref_data = SeqIO.index(ref_file, "fasta")
        else:
            self.ref_data = SeqIO.to_dict(SeqIO.parse(ref_file,"fasta"))
        self.case = 0
        # Base dictionary for choosing error bases
        self.base = ['A', 'C', 'G', 'T']
        for key in self.ref_data:
            if filter == "human" and len(self.ref_data[key].seq) > 450000:
                self.ref_data_keys.append(key)
            else:
                self.ref_data_keys.append(key)
            if self.ref_data[key].seq.count("M") > 0:
                if self.debug: print("Bases M detected in sequence")

                # Set case to 1
                self.case = 1
                # New Base Dictionary in case 1 with extra bases M, N for choosing error bases
                self.base = ['A', 'C', 'G', 'T', 'M']
            if self.ref_data[key].seq.count("N") > 0:
                self.empty_regions = True

    def load_config(self, config_file):
        print("Loading config file")
        handle = open(config_file,mode='rb')
        self.signal_config = SIConfigFile()
        self.signal_config.load_file(handle)
        lengths = []
        probabilities = []
        for rld in self.signal_config.read_length_distribution:
            lengths.append(rld[0])
            probabilities.append(rld[1])
        self.read_length_distribution.append(lengths)
        self.read_length_distribution.append(probabilities)
        lengths = []
        probabilities = []
        for eld in self.signal_config.event_length_distribution:
            lengths.append(eld[0])
            probabilities.append(eld[1])
        self.event_length_distribution.append(lengths)
        self.event_length_distribution.append(probabilities)
        lengths = []
        probabilities = []
        for offset in self.signal_config.offsets:
            lengths.append(offset[0])
            probabilities.append(offset[1])
        self.offsets.append(lengths)
        self.offsets.append(probabilities)
        lengths = []
        probabilities = []
        for range in self.signal_config.ranges:
            lengths.append(range[0])
            probabilities.append(range[1])
        self.ranges.append(lengths)
        self.ranges.append(probabilities)
        self.bases_per_second = self.signal_config.bases_per_second
        self.pores_number = self.signal_config.pores_number
        self.max_active_pores = self.signal_config.max_active_pores
        self.read_until = self.signal_config.read_until
        self.wear_out = self.signal_config.wear_out
        self.file_path = self.signal_config.file_path


    def generate(self, snip_count, generate_events=True, ticks=None, precise=False, debug=False, sampling_rate = 4000.0, cut_off_freq = 1750.0, bandwidth_freq = 40.0):
        # Generate random snippets
        distributed_length = np.random.choice(self.read_length_distribution[0], snip_count, p=self.read_length_distribution[1])
        length_offsets = np.random.randint(0, 1000, snip_count)
        snippet_length = np.maximum(distributed_length - length_offsets, np.full(snip_count, 20))
        snippet_length.sort()
        references_keys = np.random.choice(self.ref_data_keys, snip_count)
        references_lengths = np.array([len(self.ref_data[key]) for key in references_keys])
        references_indices = np.argsort(references_lengths)
        references_lengths_sorted = references_lengths[references_indices]
        snippet_starts = np.array([np.random.randint(0, max(1,ref)) for ref in references_lengths_sorted - snippet_length])
        snippets = np.array([self.ref_data[references_keys[references_indices[i]]].seq[snippet_starts[i]:min(references_lengths_sorted[i], snippet_starts[i] + snippet_length[i])] for i in range(len(references_keys))])

        reads = []

        #generate a low pass filter
        fS = sampling_rate  # Sampling rate.
        fL = cut_off_freq  # Cutoff frequency.
        fb = bandwidth_freq
        b = fb / fS
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1  # Make sure that N is odd.
        n = np.arange(N)

        # Compute sinc filter.
        h = np.sinc(2 * fL / fS * (n - (N - 1) / 2.))

        # Compute Blackman window.
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * n / (N - 1))

        h = h * w

        h /= np.sum(h)

        impulse = repeat(0., len(h));
        impulse[0] = 1.
        h_response = scipy.signal.lfilter(h, 1, impulse)
        # g_response = scipy.signal.lfilter(g,1,impulse)

        h_start = np.argmax(h_response)

        for i, seq in enumerate(snippets):
            is_reverse = np.random.randint(0, 2)
            if is_reverse:
                new_record = list(Seq.reverse_complement(seq))
            else:
                new_record = list(seq)
            # Generate Errors

            # Mismatch error
            mis_error = 0
            # Deletion error
            del_error = 0
            # Insertion error
            ins_error = 0

            # Calculate the error amount:
            amount_error = int(0 * len(new_record))
            if self.debug: print("Sequence length: ", len(new_record))
            if self.debug: print("Error amount: ", amount_error)

            # Equal distribution of all error types
            # e = int(amount_error / 3)

            # New Sequence
            sequence = ""

            # Help variables
            error_count = 0
            # i = 0

            error_sites = np.random.choice(len(new_record), amount_error,replace=False)
            error_sites.sort()

            for error_site in error_sites:

                # Random error type 1 = Mismatch, 2 = Insertion, 3 = Deletion
                error_type = randint(1, 3)

                if error_type == 1:
                    choice = sample(self.base, 2)
                    if new_record[error_site] == choice[0]:
                        # Change choice, if same base as reference
                        new_record[error_site] = choice[1]

                    else:
                        new_record[error_site] = choice[0]

                elif error_type == 2:
                    choice = sample(self.base, 2)
                    new_record.insert(error_site,choice[0])
                    error_sites[error_sites > error_site] = error_sites[error_sites > error_site] + 1
                else:
                    del(new_record[error_site])
                    error_sites[error_sites > error_site] = error_sites[error_sites > error_site] - 1

            # if in the reference not all bases are known these need to be simulated as well
            if self.empty_regions:
                new_record = np.array(new_record)
                new_record[new_record == 'N'] = np.random.choice(self.base, 1)

            sequence = "".join(new_record)

            if self.debug: print(sequence)
            if self.debug: print("Amount mismatch: %s, insertion: %s, deletion: %s" % (mis_error, ins_error, del_error))
            if self.debug: print("Generated sequence with %s errors " % (error_count))

            # Generate Events

            # Divide sequence into kmers
            kmers = [sequence[i:i + self.k] for i in range(0, len(sequence) - self.k + 1)]

            try:
                kmer_means, kmer_stdvs = zip(*[self.model_dict[kmer] for kmer in kmers])
            except ValueError:
                print("Reference too short to simulate. Continueing!")
                continue
            kmer_means = np.array(kmer_means)
            kmer_stdvs = np.array(kmer_stdvs)

            if debug:
                event_move = np.full(len(kmer_means),1)
            else:
                event_move = np.ceil(np.random.exponential(scale=2, size=len(kmer_means))).astype('int')
            event_total = np.sum(event_move)
            move_column = np.zeros(event_total, dtype='int')
            if not debug:
                move_column[np.cumsum(event_move[:-2])] = 1

            event_idx = np.repeat(np.arange(len(kmer_means)), event_move)
            event_std = np.random.uniform(-1 * kmer_stdvs[event_idx], kmer_stdvs[event_idx])
            if not precise:
                event_mean = kmer_means[event_idx] + event_std
            else:
                event_mean = kmer_means[event_idx]

            if not ticks:
                event_samples = np.random.choice(self.event_length_distribution[0], event_total, p=self.event_length_distribution[1]).astype(int)
            else:
                event_samples = np.full(event_total,ticks)
            event_list = np.stack([event_mean, np.abs(event_std), event_samples, np.array(kmers)[event_idx], move_column], axis=1)

            # Generate reads (Raw data)
            MW = []
            # For the length of an event, generate reads depending on mean
            [MW.extend((float(event[0]),) * int(event[2])) for event in event_list]
            if not precise:
                MW = np.convolve(MW,h)[h_start+1:-(N-h_start-1)+1]  # filter with low pass filter
                Noise = []
                [Noise.extend(np.random.normal(((float(event[1])) * -1/2), (2*float(event[1])/2), int(event[2]))) for event in event_list]
                MW = MW + Noise # add noise
                signal_offset = float(np.random.choice(self.offsets[0],1,p=self.offsets[1]))
                signal_range = np.random.choice(self.ranges[0],1,p=self.ranges[1])
                signal = np.ceil(MW * (8192 / signal_range) + signal_offset)

            if generate_events:
                reads.append([sequence, np.array(signal).astype(int), float(signal_offset), float(signal_range), int(snippet_starts[i]), int(snippet_length[i]), event_list])
            else:
                reads.append([sequence, np.array(signal).astype(int), float(signal_offset), float(signal_range), int(snippet_starts[i]), int(snippet_length[i])])

        return reads