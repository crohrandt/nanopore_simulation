from .SIConfigFile import SIConfigFile
from random import *
from Bio import SeqIO
from Bio import Seq
import numpy as np
from pylab import *
import scipy.signal
from datetime import date
import time
import h5py
import os
import subprocess
from io import BytesIO


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
            if filter == "human" and len(self.ref_data[key].seq) > 4500000:
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


    def generate(self, snip_count, generate_events=True, ticks=None, precise=False, debug=False, scrappie=False, length_distribution=None, offset = 3, sampling_rate = 4000.0, cut_off_freq = 1750.0, bandwidth_freq = 40.0):

        if length_distribution:
            read_length_distribution = length_distribution
        else:
            read_length_distribution = self.read_length_distribution

        # Generate random snippets
        distributed_length = np.random.choice(read_length_distribution[0], snip_count, p=read_length_distribution[1])
        length_offsets = np.random.randint(0, 1000, snip_count)
        snippet_length = np.maximum(distributed_length - length_offsets, np.full(snip_count, 20))
        snippet_length.sort()
        references_keys = np.random.choice(self.ref_data_keys, snip_count)
        references_lengths = np.array([len(self.ref_data[key]) for key in references_keys])
        references_indices = np.argsort(references_lengths)
        references_lengths_sorted = references_lengths[references_indices]
        snippet_starts = np.array([np.random.randint(0, max(1,ref)) for ref in references_lengths_sorted - snippet_length])
        snippets = [self.ref_data[references_keys[references_indices[i]]].seq[snippet_starts[i]:min(references_lengths_sorted[i], snippet_starts[i] + snippet_length[i])] for i in range(len(references_keys))]

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

            if scrappie:
                proc = subprocess.run(["scrappie squiggle <(echo -e \">\\n" + sequence + "\")"], stdout=subprocess.PIPE, executable='/bin/bash', shell=True)
                scrappie_output_array = np.genfromtxt(BytesIO(proc.stdout),dtype="i4,S1,f4,f4,f4")

            if self.debug: print(sequence)
            if self.debug: print("Amount mismatch: %s, insertion: %s, deletion: %s" % (mis_error, ins_error, del_error))
            if self.debug: print("Generated sequence with %s errors " % (error_count))

            # Generate Events

            # Divide sequence into kmers
            kmers = [sequence[i:i + self.k] for i in range(0, len(sequence) - self.k + 1)]

            try:
                kmer_means, kmer_stdvs = zip(*[self.model_dict[kmer] for kmer in kmers])
            except ValueError:
                print("Reference too short to simulate. Continuing!")
                continue
            kmer_means = np.array(kmer_means)
            kmer_stdvs = np.array(kmer_stdvs)

            if not scrappie:
                event_std = np.random.uniform(-1 * kmer_stdvs, kmer_stdvs)
                if not precise:
                    event_mean = kmer_means + event_std
                else:
                    event_mean = kmer_means
                if not ticks:
                    event_samples = np.random.choice(self.event_length_distribution[0], len(event_mean), p=self.event_length_distribution[1]).astype(int)
                else:
                    event_samples = np.full(len(event_mean),ticks)
                event_list = np.stack([event_mean, np.abs(event_std), event_samples, np.array(kmers)], axis=1)
            else:
                event_std = np.random.uniform(-1 * kmer_stdvs, kmer_stdvs)
                if not precise:
                    event_mean = kmer_means + event_std
                else:
                    event_mean = kmer_means
                event_samples = np.array([int(np.round(np.random.geometric(1/length,1)))+offset for length in scrappie_output_array['f4'][:-(self.k - 1)]])
                event_list = np.stack([event_mean, np.abs(event_std), event_samples, np.array(kmers)], axis=1)

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


    def generate_fast5_worker(self, config_file, simulation_queue, counter, config, read_length):
        while not simulation_queue.empty():
            file = simulation_queue.get()
            path = file[0]
            read_number = int(file[1])
            start_time = file[2]

            if 'sampling_rate' in config:
                sampling_rate = config['MinION Configuration']['sample_rate']
            else:
                sampling_rate = config_file.sampling_rate

            read = self.generate(1, sampling_rate=sampling_rate, scrappie=config['Simulation Parameters'].getboolean('use_scrappie'), cut_off_freq=float(config['Low Pass Filter'].get('cut_off_frequency','1750.0')), bandwidth_freq=float(config['Low Pass Filter'].get('bandwidth_frequency','40.0')), length_distribution=read_length, offset=int(config['Simulation Parameters'].get('scrappie_length_offset', 3)))[0]
            while type(read).__name__ != 'list':
                read = self.generate(1, sampling_rate=sampling_rate, scrappie=config['Simulation Parameters'].getboolean('use_scrappie'), cut_off_freq=float(config['Low Pass Filter'].get('cut_off_frequency', '1750.0')), bandwidth_freq=float(config['Low Pass Filter'].get('bandwidth_frequency', '40.0')), length_distribution=read_length, offset=int(config['Simulation Parameters'].get('scrappie_length_offset', 3)))[0]

            values_offset = read[2]
            values_range = read[3]
            signal = read[1]
            channel_number = file[3]
            hostname = "Simulator"
            date_string = date.today().strftime("%Y%m%d")
            flowcell = "FL_Sim"
            purpose = "sim"
            device = "Nanopore_SimulatION"
            sample_id = config_file.sample_id + "_simulated"

            if not os.path.exists(path):
                os.makedirs(path)

            filename = hostname + "_" + date_string + "_" + flowcell + "_" + device + "_" + purpose + "_" + sample_id + "_75432" + "_ch" + str(
                channel_number) + "_read" + str(read_number) + "_strand.fast5"

            f = h5py.File(os.path.join(path, filename), 'w')
            f.attrs.create("file_version", data=0.6, dtype="float64")

            grp = f.create_group("UniqueGlobalKey/channel_id")
            grp.attrs.create("channel_number", data=channel_number, dtype="S3")
            grp.attrs.create("digitisation", data=config_file.digitisation, dtype="float64")
            grp.attrs.create("offset", data=values_offset, dtype="float64")
            grp.attrs.create("range", data=values_range, dtype="float64")
            grp.attrs.create("sampling_rate", data=config_file.sampling_rate, dtype="float64")

            grp = f.create_group("UniqueGlobalKey/context_args")
            grp.attrs.create("experiment_kit", data=np.string_(config_file.experiment_kit))
            grp.attrs.create("filename", data=np.string_(filename))
            grp.attrs.create("sample_frequency", data=np.string_(str(config_file.sampling_rate)))
            grp.attrs.create("user_filename_input", data=np.string_(sample_id))

            grp = f.create_group("UniqueGlobalKey/tracking_id")
            grp.attrs.create("asic_id", data=np.string_(config_file.asic_id))
            grp.attrs.create("asic_id_eeprom", data=np.string_(config_file.asic_id_eeprom))
            grp.attrs.create("asic_temp", data=np.string_(str(
                np.random.choice(config_file.get_distribution_keys(config_file.asic_temp), 1,
                                 p=config_file.get_distribution_probabilities(config_file.asic_temp))[0])))
            grp.attrs.create("auto_update", data=np.string_(config_file.auto_update))
            grp.attrs.create("auto_update_source", data=np.string_(config_file.auto_update_source))
            if config_file.bream_core_version != 'None':
                grp.attrs.create("bream_core_version", data=np.string_(config_file.bream_core_version))
            if config_file.bream_is_standard != 'None':
                grp.attrs.create("bream_is_standard", data=np.string_(config_file.bream_is_standard))
            if config_file.bream_nc_version != 'None':
                grp.attrs.create("bream_nc_version", data=np.string_(config_file.bream_nc_version))
            if config_file.bream_ont_version != 'None':
                grp.attrs.create("bream_ont_version", data=np.string_(config_file.bream_ont_version))
            if config_file.bream_prod_version != 'None':
                grp.attrs.create("bream_prod_version", data=np.string_(config_file.bream_prod_version))
            if config_file.bream_rnd_version != 'None':
                grp.attrs.create("bream_rnd_version", data=np.string_(config_file.bream_rnd_version))
            grp.attrs.create("device_id", data=np.string_(device))
            grp.attrs.create("exp_script_name", data=np.string_(config_file.exp_script_name))
            grp.attrs.create("exp_script_purpose", data=np.string_(purpose))
            grp.attrs.create("exp_start_time", data=np.string_(str(start_time)))
            grp.attrs.create("flow_cell_id", data=np.string_(flowcell))
            grp.attrs.create("heatsink_temp", data=np.string_(str(
                np.random.choice(config_file.get_distribution_keys(config_file.heatsink_temp), 1,
                                 p=config_file.get_distribution_probabilities(config_file.heatsink_temp))[
                    0])))
            grp.attrs.create("hostname", data=np.string_(hostname))
            grp.attrs.create("installation_type", data=np.string_(config_file.installation_type))
            grp.attrs.create("local_firmware_file", data=np.string_(config_file.local_firmware_file))
            grp.attrs.create("operating_system", data=np.string_(config_file.operating_system))
            grp.attrs.create("protocol_run_id", data=np.string_(config_file.protocol_run_id))
            grp.attrs.create("protocols_version", data=np.string_(config_file.protocols_version))
            grp.attrs.create("run_id", data=np.string_(config_file.run_id))
            grp.attrs.create("sample_id", data=np.string_(sample_id))
            grp.attrs.create("usb_config", data=np.string_(config_file.usb_config))
            grp.attrs.create("version", data=np.string_(config_file.version_number))

            grp = f.create_group("Raw/Reads/Read_" + str(read_number))
            grp.attrs.create("duration", data=len(signal), dtype="int32")
            grp.attrs.create("median_before", data=250, dtype="float64")
            grp.attrs.create("read_id", data=np.string_("16acf7fb-696b-4b96-b95b-0a43f" + format(read_number, '07d')))
            grp.attrs.create("read_number", data=read_number, dtype="int32")
            grp.attrs.create("start_mux", data=1, dtype="int32")
            grp.attrs.create("start_time", data=int(time.time()), dtype="int64")

            grp.create_dataset("Signal", data=signal.astype("int16"), dtype="int16", compression="gzip",
                               compression_opts=1, maxshape=(None,))

            f.close()
            with counter.get_lock():
                counter.value += 1
