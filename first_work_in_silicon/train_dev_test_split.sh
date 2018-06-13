#/bin/bash
sort -R data/cut.csv > data/sort.csv
head -n 7478 data/sort.csv > data/train.csv
tail -n 512 data/sort.csv > data/tmp.csv
head -n 256 data/tmp.csv > data/dev.csv
tail -n 256 data/tmp.csv > data/test.csv
rm data/tmp.csv
rm data/sort.csv