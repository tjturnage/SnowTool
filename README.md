# SnowTool
Python 3

graphical displays of Cobb Snow Tool output
Downloads files from Penn State example ( http://www.meteo.psu.edu/bufkit/data/HRRR/00/hrrr_kgrr.cobb )
Plots time series of snow accumulations using Pandas

Requires the following local directories be created and accessible for handling text files

- raw/          : downloaded file location
- processed/    : after text in downloaded files is reformatted the results are written into files in this directory
-                  file format is **YYYYMMDD_HH_{model}_{station).txt** and refers to model run times
- stage/        : staging location for selected processed files that will be read into dataframes for plotting
