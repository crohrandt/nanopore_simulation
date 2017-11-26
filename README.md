## Nanopore SimulatION

| Inputs | Outputs |
|---|---|
| Configuration-File (derived form a real conducted experiment)| 
| Reference genome FASTA-File (may be from another species)| Basecallable Fast5-Files with raw values |
| Model-File (provided by ONT)| |

### Description

Nanopore SimulatION is a tool for simulating an Oxford Nanopore Technologies MinION device for bioinformatic development. 

### Installation

For Debian-based distros (Debian, Ubuntu, Linux Mint) install:
```bash
sudo apt-get install python3-tk
```

For CentOS, Redhat Linux install:
```bash
sudo yum install python3-tkinter
```

After that, install the Nanopore SimulatION package:

```bash
git clone https://github.com/crohrandt/nanopore_simulation
cd nanopore_simulation
pip3 install -e ./
```

##### Dependencies

All dependencies should be automatically installed by pip.

###### Installation
- numpy
- scipy
- biopython
- argparse
- pandas
- h5py
- matplotlib (for future use, yet only pylab is used)

### Usage

#### Examples

In the examples directory several standard use cases will be predefined for quick usage. Each use 
case is structured within an own subdirectory. To try out the examples the bash-script needs 
to be run. Every file that is requireed is packaged or will be downloaded from a free and open 
source download site.

For the complete verification, a pipeline is described using the ONT albacore basecaller in version 2.1.1 and a mapping 
of the simulated reads to the reference genome using minimap2 in version 2.1-r311. 

#### DNA-Example

###### Simulate human DNA reads
```
cd examples
cd DNA
./DNA_quick_example_run.sh
```
or
```
./DNA_full_example_run.sh
```
###### Basecall simulated Fast5s with albacore 2.1.1
```
cd Run-Output
read_fast5_basecaller.py -i . -o fast5,fastq -s basecalled -f FLO-MIN106 -k SQK-LSK108 -t 4 -r
```

###### Map basecalled reads with minimap 2.1

```
minimap2 -ax map-ont ../../Homo_sapiens.GRCh38.dna.primary_assembly.fa basecalled/workspace/pass/*.fastq > Run-Output.sam
```

###### Tested with

- ONT albacore 2.1.1
- minimap2 2.11-r311
