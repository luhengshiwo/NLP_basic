#/bin/bash
sort -R data/cut.csv > data/sort.csv
head -n 9216 data/sort.csv > data/train.csv
tail -n 1024 data/sort.csv > data/tmp.csv
head -n 512 data/tmp.csv > data/dev.csv
tail -n 512 data/tmp.csv > data/test.csv
rm data/tmp.csv
rm data/sort.csv
