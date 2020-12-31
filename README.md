# SnowTool
## Description:

Plots time series of Cobb Snow Tool text output. These files are routinely created at the [Penn State Bufkit site](http://www.meteo.psu.edu/bufkit/)


[Cobb Bufkit Data Distribution System webpage](http://www.meteo.psu.edu/bufkit/CONUS_HRRR_00_cobb.html)


Example of file that's downloaded: <http://www.meteo.psu.edu/bufkit/data/HRRR/00/hrrr_kgrr.cobb>


### Requires the following local directories be created and accessible for handling text files 

|    directory    |  description                                                                                                                           |
| ---------------:|:-------------------------------------------------------------------------------------------------------------------------------------- | 
|       `raw/`    |    downloaded files dir                                                                                                                |
|   `processed/`  |    destination for files parsed from `raw/` and file format is `YYYYMMDD_HH_{model}_{station).txt` based on model run time             |
|      `stage/`   |    based on desired stations, models, and number of model runs to plot the associated files get staged here                            |

