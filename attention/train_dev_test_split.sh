#/bin/bash
sort -R data/cut.csv > data/sort.csv
head -n 35249 data/sort.csv > data/train.csv
tail -n 4096 data/sort.csv > data/tmp.csv
head -n 2048 data/tmp.csv > data/dev.csv
tail -n 2048 data/tmp.csv > data/test.csv
rm data/tmp.csv
rm data/sort.csv