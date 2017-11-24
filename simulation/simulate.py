from simulation.SimSignalGenerator import SimSignalGenerator
import numpy as np
import h5py
import random
from datetime import date
import time
import os


def run(parser, args):

    my_generator = SimSignalGenerator(config_file=args.config)
    my_generator.load_reference(ref_file=args.ref,filter=args.filter)
    my_generator.load_model(args.model)

    start_time = int(time.time())
    (N, n) = divmod(args.reads,args.dirreads)

    directory = os.path.join("./",args.output)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for k in range(N + 1):
        if k == N:
            if n == 0:
                break
            number_reads = n
        else:
            number_reads = args.dirreads

        new_directory = os.path.join(directory, str(k))
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        j = 0

        (M, m) = divmod(number_reads,args.workerreads)

        for j in range(M+1):
            if j == M:
                sub_number_reads = m
            else:
                sub_number_reads = args.workerreads

            reads = my_generator.generate(sub_number_reads,sampling_rate=my_generator.signal_config.sampling_rate,cut_off_freq=args.cut_off,bandwidth_freq=args.band)
            random.shuffle(reads)

            for i, read in enumerate(reads):
                values_offset = read[2]
                values_range = read[3]
                sampling_rate = my_generator.signal_config.sampling_rate
                signal = read[1]
                read_number = k*args.dirreads + j*args.workerreads + i + 1
                channel_number = random.randint(1, 512)
                hostname = "Simulator"
                date_string = date.today().strftime("%Y%m%d")
                flowcell = "FL_Sim"
                purpose = "sim"
                device = "Nanopore_SimulatION"
                sample_id = my_generator.signal_config.sample_id + "_simulated"
                filename = hostname + "_" + date_string + "_" + flowcell + "_" + device + "_" + purpose + "_" + sample_id + "_75432" + "_ch" + str(
                    channel_number) + "_read" + str(read_number) + "_strand.fast5"

                f = h5py.File(os.path.join(new_directory, filename), 'w')
                f.attrs.create("file_version", data=0.6, dtype="float64")

                grp = f.create_group("UniqueGlobalKey/channel_id")
                grp.attrs.create("channel_number", data=channel_number, dtype="S3")
                grp.attrs.create("digitisation", data=my_generator.signal_config.digitisation, dtype="float64")
                grp.attrs.create("offset", data=values_offset, dtype="float64")
                grp.attrs.create("range", data=values_range, dtype="float64")
                grp.attrs.create("sampling_rate", data=my_generator.signal_config.sampling_rate, dtype="float64")

                grp = f.create_group("UniqueGlobalKey/context_args")
                grp.attrs.create("experiment_kit", data=np.string_(my_generator.signal_config.experiment_kit))
                grp.attrs.create("filename", data=np.string_(filename))
                grp.attrs.create("sample_frequency", data=np.string_(str(my_generator.signal_config.sampling_rate)))
                grp.attrs.create("user_filename_input", data=np.string_(sample_id))

                grp = f.create_group("UniqueGlobalKey/tracking_id")
                grp.attrs.create("asic_id", data=np.string_(my_generator.signal_config.asic_id))
                grp.attrs.create("asic_id_eeprom", data=np.string_(my_generator.signal_config.asic_id_eeprom))
                grp.attrs.create("asic_temp", data=np.string_(str(np.random.choice(
                    my_generator.signal_config.get_distribution_keys(my_generator.signal_config.asic_temp), 1,
                    p=my_generator.signal_config.get_distribution_probabilities(my_generator.signal_config.asic_temp))[
                                                                      0])))
                grp.attrs.create("auto_update", data=np.string_(my_generator.signal_config.auto_update))
                grp.attrs.create("auto_update_source", data=np.string_(my_generator.signal_config.auto_update_source))
                if my_generator.signal_config.bream_core_version != 'None':
                    grp.attrs.create("bream_core_version",
                                     data=np.string_(my_generator.signal_config.bream_core_version))
                if my_generator.signal_config.bream_is_standard != 'None':
                    grp.attrs.create("bream_is_standard", data=np.string_(my_generator.signal_config.bream_is_standard))
                if my_generator.signal_config.bream_nc_version != 'None':
                    grp.attrs.create("bream_nc_version", data=np.string_(my_generator.signal_config.bream_nc_version))
                if my_generator.signal_config.bream_ont_version != 'None':
                    grp.attrs.create("bream_ont_version", data=np.string_(my_generator.signal_config.bream_ont_version))
                if my_generator.signal_config.bream_prod_version != 'None':
                    grp.attrs.create("bream_prod_version",
                                     data=np.string_(my_generator.signal_config.bream_prod_version))
                if my_generator.signal_config.bream_rnd_version != 'None':
                    grp.attrs.create("bream_rnd_version", data=np.string_(my_generator.signal_config.bream_rnd_version))
                grp.attrs.create("device_id", data=np.string_(device))
                grp.attrs.create("exp_script_name", data=np.string_(my_generator.signal_config.exp_script_name))
                grp.attrs.create("exp_script_purpose", data=np.string_(purpose))
                grp.attrs.create("exp_start_time", data=np.string_(str(start_time)))
                grp.attrs.create("flow_cell_id", data=np.string_(flowcell))
                grp.attrs.create("heatsink_temp", data=np.string_(str(np.random.choice(
                    my_generator.signal_config.get_distribution_keys(my_generator.signal_config.heatsink_temp), 1,
                    p=my_generator.signal_config.get_distribution_probabilities(
                        my_generator.signal_config.heatsink_temp))[0])))
                grp.attrs.create("hostname", data=np.string_(hostname))
                grp.attrs.create("installation_type", data=np.string_(my_generator.signal_config.installation_type))
                grp.attrs.create("local_firmware_file", data=np.string_(my_generator.signal_config.local_firmware_file))
                grp.attrs.create("operating_system", data=np.string_(my_generator.signal_config.operating_system))
                grp.attrs.create("protocol_run_id", data=np.string_(my_generator.signal_config.protocol_run_id))
                grp.attrs.create("protocols_version", data=np.string_(my_generator.signal_config.protocols_version))
                grp.attrs.create("run_id", data=np.string_(my_generator.signal_config.run_id))
                grp.attrs.create("sample_id", data=np.string_(sample_id))
                grp.attrs.create("usb_config", data=np.string_(my_generator.signal_config.usb_config))
                grp.attrs.create("version", data=np.string_(my_generator.signal_config.version_number))

                grp = f.create_group("Raw/Reads/Read_" + str(read_number))
                grp.attrs.create("duration", data=len(signal), dtype="int32")
                grp.attrs.create("median_before", data=250, dtype="float64")
                grp.attrs.create("read_id",
                                 data=np.string_("16acf7fb-696b-4b96-b95b-0a43f" + format(read_number, '07d')))
                grp.attrs.create("read_number", data=read_number, dtype="int32")
                grp.attrs.create("start_mux", data=1, dtype="int32")
                grp.attrs.create("start_time", data=int(time.time()), dtype="int64")

                grp.create_dataset("Signal", data=signal.astype("int16"), dtype="int16", compression="gzip",
                                   compression_opts=1, maxshape=(None,))

                f.close()
            print("Simulated " + str(read_number) + " of " + str(args.reads) + " reads so far...")