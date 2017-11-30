#!/bin/bash

if [ ! -f ../Homo_sapiens.GRCh38.dna.primary_assembly.fa ]; then
    wget ftp://ftp.ensembl.org/pub/release-90/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    gunzip -c Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz > ../Homo_sapiens.GRCh38.dna.primary_assembly.fa
fi
if [ -d Run-Output ]; then
    rm -r Run-Output
fi

simulatION simulate -r ../Homo_sapiens.GRCh38.dna.primary_assembly.fa -c gDNAWA01.sic -m template_median68pA.model -n 1000 -o Run-Output -t 4
