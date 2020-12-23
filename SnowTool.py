"""
Extracting AFDs and sorting by time
"""


import re
import urllib

#import matplotlib.dates as mdates
#from matplotlib.dates import DateFormatter
#import matplotlib.gridspec as gridspec
#import seaborn as sns
from matplotlib import rcParams, cycler
import numpy as np
from matplotlib.lines import Line2D
import itertools, operator

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime,timedelta
import os, sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt


#http://www.meteo.psu.edu/bufkit/data/GFS/12/gfs3_krqb.cobb
#http://www.meteo.psu.edu/bufkit/data/NAMNEST/12/namnest_krqb.cobb
#http://www.meteo.psu.edu/bufkit/data/NAM/12/nam_kgrr.cobb
#http://www.meteo.psu.edu/bufkit/data/RAP/03/rap_kgrr.cobb
#http://www.meteo.psu.edu/bufkit/data/HRRR/22/hrrr_kepo.cobb

try:
    os.listdir('/usr')
    scripts_dir = '/data/scripts'
    image_dir = os.path.join('/var/www/html','images','snowtool')
except:
    scripts_dir = 'C:/data/scripts'
    sys.path.append(os.path.join(scripts_dir,'resources'))
    image_dir = os.path.join('C:/data/scripts')

class SnowTool:

    def __init__(self, station, model_list):
        # the issuing office, not the TAF site
        self.station = station
        self.model_list = model_list
        self.text_file = None
        self.df = None
        self.df_master = None

        
        self.made_master = False
        self.plotted = False
        self.col = 1
        pre_now = datetime.utcnow()
        self.now = pre_now.replace(minute=0,second=0,microsecond=0)
        self.base_dir = os.path.join(scripts_dir,'text')
        self.raw_dir = os.path.join(self.base_dir,'raw')
        self.processed_dir = os.path.join(self.base_dir,'processed')  
        self.staged_dir = os.path.join(self.base_dir,'staged')

        self.column_names = []
        self.legends = []
        self.staged_file_dict = {}

        self.df_master = None
        for sf in os.listdir(self.raw_dir):
            os.remove(os.path.join(self.raw_dir,sf))
        for st in os.listdir(self.staged_dir):
            os.remove(os.path.join(self.staged_dir,st))
        self.colsp = 0.4
        self.model_info = {'gfs3':{'name':'gfs3','cmap':plt.cm.Greys, 'cspace': self.colsp, 'width': 1},
                      'nam':{'name':'nam','cmap':plt.cm.Blues, 'cspace': self.colsp, 'width': 1},
                      'namnest':{'name':'nam','cmap':plt.cm.Oranges, 'cspace': self.colsp, 'width': 1},
                      'hrrr':{'name':'hrrr','cmap':plt.cm.Greens, 'cspace': self.colsp, 'width': 1},    
                      'rap':{'name':'rap','cmap':plt.cm.Purples, 'cspace': self.colsp, 'width': 1}}
        self.main()


    def main(self):
        for model in self.model_list:
            self.model = model
            self.possible_files()       # build filenames to check for locally 
            self.get_files()            # downloads files determined to not already exist
            self.get_dst_filenames()    # determines model init times by fi
            self.format_files()     # reformats to csv and writes to processed dir
            self.stage_files()      # stages processed files matching desired stn, model
            self.make_dataframe()   # makes dataframe for each model run, appends to df_master
        self.final_plot()       # plots df_master with accumulated models
        
    def possible_files(self):
        """
        Guesses which model runs would be in any files downloaded now and therefore the final
        filenames assocated with these downloads. 
        
        Filename format -- YYYYMMDD_HH_{model}_{station}.txt
            
        Returns
        -------
        creates a list of files called "self.psbl_files"

        """
        self.hrs = []
        self.psbl_files = []
        round_down_6hrs = (self.now.hour - 2)%6 + 2
        round_up_6hrs = 6 - (self.now.hour)%6
        goback = self.now - timedelta(hours=2)
        goback6 = self.now - timedelta(hours=round_down_6hrs)
        self.roundup6 = self.now + timedelta(hours=round_up_6hrs)
        self.gfs_start = self.roundup6 + timedelta(days=0)
        self.gfs_end = self.roundup6 + timedelta(hours=84)
        self.short_end = self.now + timedelta(hours=48)
        #clean = goback.replace(minute=0, second=0, microsecond=0)
        # hourly run time interval for these models, could go > 4 versions back

        #if self.model == 'hrrr' or self.model == 'rap':
        if self.model == 'blorg' or self.model == 'blah':
            
            for h in range (0,6):
                new = goback - timedelta(hours=h)
                hr = datetime.strftime(new, '%H')
                tf = datetime.strftime(new, '%Y%m%d_%H')
                psbl_file = '{}_{}_{}.txt'.format(tf,self.model,self.station)
                self.psbl_files.append(psbl_file)
            self.hrs.append(hr)
            
        #elif self.model == 'nam' or self.model == 'namnest' or self.model == 'gfs3':
        elif self.model in ['nam', 'namnest', 'rap', 'hrrr', 'gfs3']:
            # 6 hourly run time interval for these models
            for i in range (0,3):
                new = goback6 - timedelta(hours=i*6)
                hr = datetime.strftime(new, '%H')
                tf = datetime.strftime(new, '%Y%m%d_%H')
                psbl_file = '{}_{}_{}.txt'.format(tf,self.model,self.station)
                self.psbl_files.append(psbl_file)
        else:
            print('model not found!')

        #print(self.psbl_files)
        return
    

    def get_files(self):
        """
        The list of possible files created in the possible_files method is compared against
        a listing of files in the "processed" directory. Any match means no new download needed.
        
        Returns
        -------
        Writes downloaded files to raw directory

        """
        self.downloaded_files = []
        self.already = os.listdir(self.processed_dir)

        for t in self.psbl_files:
            if t in self.already:
                if self.station in t and self.model in t:
                    print('already exists: ' + str(t) )
                pass
            else:
                print('retrieving ... ' + str(t) )
                t2 = t.replace('txt','cobb')
                #20201219_12_gfs3_kmkg.cobb
                els = t2.split('_')[1:]
                #['12', 'gfs3', 'kmkg.cobb']
                hr = els[0]
                loc = els[2]
                if self.model == 'gfs3':
                    modup = 'GFS'
                else:
                    modup = self.model.upper()
                    
                #GFS/12/gfs3_krqb.cobb
                uri = '{}/{}/{}_{}'.format(modup,hr,self.model,loc)
                fout = '{}_{}_{}'.format(hr,self.model,loc)
                #http://www.meteo.psu.edu/bufkit/data/GFS/12/gfs3_krqb.cobb
                url = 'http://www.meteo.psu.edu/bufkit/data/' + uri

                fpath = os.path.join(self.raw_dir,fout)
 
                try:
                    print('downloading ... ' + url ) 
                    response = urllib.request.urlopen(url)
                    webContent = response.read()
                    f = open(fpath, 'wb')
                    f.write(webContent)
                    f.close()
                    self.downloaded_files.append(fout)
                except:
                    print('could not download: ' + t)

        return

    def get_dst_filenames(self):
        """
        Goes through newly downloaded files to find the string that indicates the model init time.
        This informs what the destination filenames should be after these raw files get processed.

        Returns
        -------
        None.

        """
        self.dst_fname_list = []
        find_dt = re.compile('\d{8}/\d{4}')
        for f in self.downloaded_files:
            src_file = os.path.join(self.raw_dir,f)
            with open(src_file) as src:
                for line in src:
                    m = find_dt.search(line)
                    if m is not None:
                        issued = datetime.strptime(m[0],'%Y%m%d/%H%M')
                        ftime = datetime.strftime(issued, '%Y%m%d_%H')
                        fname = "{}_{}_{}.txt".format(ftime,self.model,self.station)
                        #dst_path = os.path.join(self.processed_dir,fname)
                        self.dst_fname_list.append(fname)

                    else:
                        pass
        
        return


    def format_files(self):
        """
        Using a list of the downloaded files and a list of the destination filenames that will be 
        written to with the processed data.

        Returns
        -------
        None.

        """

        for s,d in zip(self.downloaded_files,self.dst_fname_list):
            self.temp = ''
            src_file = os.path.join(self.raw_dir,s)

            with open(src_file, 'r') as src:
                for line in src:

                    if self.model in ('nam','namnest','rap','hrrr','gfs3'):

                        if 'FH' in line and 'FHR' not in line:                                
                            t = line[2:]
                            t2 = t.replace('|', ' ')
                            t3 = t2.replace('          ', ' nowx ')
                            t4 = t3.replace(',', ' ')
                            t5 = t4.split()
                            self.temp = self.temp + ', '.join(t5) + '\n'

                    else:
                        print('model not recognized!')

            dst_file = os.path.join(self.processed_dir,d)
            with open(dst_file, 'w') as dst:
                for line in self.temp.splitlines():
                    dst.write(line + '\n')
         
        return

    def add_to_staged_file_dict(self):
        find_dts = re.compile('\d{8}_\d{2}')        
        dts_match = find_dts.search(self.text_file)
        dts = dts_match[0]
        model_run_time = datetime.strptime(dts, '%Y%m%d_%H')
        df_column_name_time_substring = datetime.strftime(model_run_time, '_%d%H')
        filename_segments = self.text_file.split('_')
        model_name = filename_segments[2]
        column_substring = model_name + df_column_name_time_substring
        self.staged_file_dict[self.text_file] = {'model': model_name,
                                                 'issued': model_run_time,
                                                 'colname': column_substring                                          
                                                 }
        return


    def stage_files(self):
        """
        Based on the station and model associated with this object instance, we can
        copy the pertinent files from the processed directory into the staging directory.

        """
        
        self.model_count = 0
        self.already = sorted(os.listdir(self.processed_dir),reverse = True)
        for f in self.already:
            if self.station in f and self.model in f and self.model_count < 2:
                if int(f.split('_')[1])%6 == 0 :
                    self.text_file = f
                    self.add_to_staged_file_dict()
                    src = os.path.join(self.processed_dir,f)
                    dst = os.path.join(self.staged_dir,f)
                    shutil.copy(src, dst)
                    self.model_count = self.model_count + 1
            else:
                pass

    def make_dataframe(self):
        """
        Read through all the processed csv files that were copied into the staging directory. Initializes
        a master dataframe (df_master) so that all processed files can append their snowfall columns to it.
        A weighted column is created in df_master to do a weighted average 

        Returns
        -------
        None.

        """

        for f in os.listdir(self.staged_dir):
            column_name_substr = self.staged_file_dict[f]['colname']
            fname = os.path.join(self.staged_dir,f)
            fhrs = None
            self.df = None

            cols = ['FH', 'Day' ,'Mon', 'date' ,'hour', 'Wind','SfcT','Ptype','Snow','TotSN','ObsSN','Sleet',
                'TotPL','FZRA','TotZR','QPF','TotQPF','SRat','CumSR','ObsSR','PcntSN','PcntPL','PcntRA','pcpPot','pthick']

            self.df = pd.read_csv(fname, names=cols)

            # remove 'KT' from Wind column strings, extract wspd string, convert to ints            
            self.df['Wind'] = [int(str(wspd)[-4:-2]) for wspd in self.df['Wind']]
            # remove 'F' from Tsfc column strings, convert to floats     
            self.df['SfcT'] = [float(str(Temp)[:-1]) for Temp in self.df['SfcT']]
    
            fhrs = list(self.df.FH.values)
            dts = []
            for fh in range(0,len(fhrs)):
                hrs = int(fhrs[fh])
                dt = self.staged_file_dict[f]['issued'] + timedelta(hours=hrs)
                dts.append(dt)

            self.df['Datetime'] = dts

            self.df = self.df.set_index(pd.DatetimeIndex(self.df['Datetime']))
            self.df = self.df.rename_axis(None)

            # initialize master dataframe if not done yet
            if self.df_master is None:
                self.df_master = self.df

            for c,p in zip(['Snow','FZRA','Sleet','SfcT','Wind'], ['snow_', 'fzra_', 'sleet_', 'temp_', 'wind_']):
            
                x = self.df[c].astype(float)
                this_column_name = p + column_name_substr
                self.df_master[str(this_column_name)] = x
                if this_column_name not in self.column_names:
                    self.column_names.append(this_column_name)            

            
            # these are "original" columns consisting of accumulations            
            for c,a in zip(['Snow','FZRA','Sleet'], ['acsn_', 'aczr_', 'acpl_']):
                this_column_name = a + column_name_substr
                self.df_master[str(this_column_name)] = self.df[c].cumsum()
                if this_column_name not in self.column_names:
                    self.column_names.append(this_column_name)                  

        return

    def plot_model(self):

        els = self.n.split('_')
        model = els[1]
        runtime = els[-1]
        if len(model) > 5:
            mod_leg = 'nest'
        else:
            mod_leg = model
        legend_str = '{}_{}'.format(mod_leg,runtime)
        self.legends.append(legend_str)
        cmap = self.model_info[model]['cmap']
        c = self.model_info[model]['cspace']
        rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 4)))
        print(self.n, self.df_master[self.n])
        plt.plot(self.df_master[self.n], color=cmap(c), linewidth=3)
        self.custom_line = Line2D(self.df_master[self.n], [0], color=cmap(c), lw=c*5)

        self.custom_line_list.append(self.custom_line)
        self.model_info[model]['cspace'] = self.model_info[model]['cspace'] + (1 - self.colsp)/4
        return


    def remove_plot_junk(self):
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)      
        self.ax.spines['right'].set_visible(False)
        #self.ax.get_xaxis.tick_bottom()
        #self.ax.get_yaxis.tick_left()
        #self.plt.yticks(fontsize=10)
        self.ax.grid(linestyle='--', alpha=0.5)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=True, left=False, right=False, labelleft=True)
        
        return


    def final_plot(self):
        numrows = 2
        self.fig, self.axes = plt.subplots(nrows=numrows,sharex=True,figsize=(10,4))        
        self.custom_line_list = []
        for self.n in self.column_names:
            if 'temp' in self.n:
                self.plot_model()

        self.ax.legend(self.custom_line_list, self.legends, fontsize=8)

        props = dict(boxstyle='round', facecolor='white', alpha=0.7)

        title_str = 'Snow Accum\nstation: {}'.format(self.station)
        self.ax.text(0.23, 0.95, title_str, transform=self.ax.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)

        self.remove_plot_junk()

        self.ax.set_xlim(pd.Timestamp(self.now), pd.Timestamp(self.short_end))            
        #self.ax.xaxis.set_major_locator(hours)
        #self.ax.xaxis.set_major_formatter(myFmt)
        img_fname = '{}.png'.format(self.station)
        image_dst_path = os.path.join(image_dir,img_fname)
        plt.savefig(image_dst_path,format='png')
        plt.close()

        return



#gfk = SnowTool('kgfk','hrrr')
#mqt = SnowTool('kmqt','gfs3')
#tlh = SnowTool('ktlh','hrrr')
#biv = SnowTool('biv','gfs3')
#grr = SnowTool('kgrr',['hrrr','rap'])#['gfs3','nam'])
olu = SnowTool('kolu',['gfs','nam','namnest','rap','hrrr'])
#mbl = SnowTool('kmbl','nam')