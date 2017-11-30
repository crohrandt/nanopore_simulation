from simulation.SimSignalGenerator import SimSignalGenerator
import math
import random
import time
import os
import sys
from multiprocessing import Process, Queue, Value, Event
import configparser
import subprocess

# print progess on command line
def process_monitor(counter, max_count, stop_event):
    import time
    try:
        while not stop_event.is_set():
            time.sleep(1)
            print("\rProcessed ", counter.value, 'of ', (max_count), end=" ")
    except KeyboardInterrupt:
        return
    except:
        print('Unknown Exception in Montior Process')


def run(parser, args):

    config = configparser.ConfigParser()
    config.read(args.ini)

    my_generator = SimSignalGenerator(config_file=args.config)
    if 'Simulation Parameters' in config:
        my_generator.load_reference(ref_file=args.ref,filter=config['Simulation Parameters'].get('filter',None))
    my_generator.load_model(args.model)

    start_time = int(time.time())

    worker = []
    simulation_queue = Queue()
    counter = Value('i', 0)
    stop_event = Event()

    read_length = []
    if 'read_length' in config['Run Parameters']:
        read_length.append([int(rlen) for rlen in config.get('Run Parameters', 'read_length').split('\n')])
        read_length.append([float(lp) for lp in config.get('Run Parameters', 'read_length_probability').split('\n')])
        if not math.isclose(sum(read_length[1]),1.0):
            print("ERROR:\t Read length distribution form ini file does not add up to 1. Exiting!")
            sys.exit(-1)

    if 'use_scrappie' in config['Simulation Parameters']:
        proc = subprocess.run(["scrappie squiggle"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='/bin/bash', shell=True)
        if proc.returncode != 64:
            print("ERROR:\t scrappie is either not installed or not in the path. Plaese install ONT scrappie (https://github.com/nanoporetech/scrappie) \n\t\t and copy/link the executable to a directory within the path. Exiting!")
            sys.exit(-1)


    for i in range(args.reads):
        simulation_queue.put([os.path.join(args.output, str(i//int(config['DEFAULT']['reads_per_dir']))), i, start_time, random.randint(1, 512)])

    monitor = Process(target=process_monitor, args=(counter, args.reads, stop_event))
    monitor.start()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    try:
        for list in range(args.threads):
            worker.append(Process(target=my_generator.generate_fast5_worker, args=(my_generator.signal_config,simulation_queue,counter,config,read_length)))
        for w in worker:
            w.start()
        for w in worker:
            w.join()

    except KeyboardInterrupt:
        pass
    stop_event.set()

    try:
        monitor.join()
        print("\n")
    except RuntimeError:
        print('Could not join monitor thread')