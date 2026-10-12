# Data

`raw/compas-scores-two-years.csv` is downloaded automatically by any
script on first use from

https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

(ProPublica, 2016; obtained by public records request from the Broward
County Sheriff's Office). `processed/df_main.csv` is the filtered analysis
sample (Section 3.1) written by `src/01_data_prep.py`. Neither file is
versioned; delete `processed/df_main.csv` to rebuild it.
