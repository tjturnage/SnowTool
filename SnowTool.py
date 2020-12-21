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

    def __init__(self, station, model_list, download=True, plot=True):
        # the issuing office, not the TAF site
        self.station = station
        self.model_list = model_list
        self.df = None
        self.df_master = None
        self.fig, self.ax = plt.subplots(1,1,sharex='row',figsize=(10,4))
        
        self.made_master = False
        self.plotted = False
        self.col = 1
        pre_now = datetime.utcnow()
        self.now = pre_now.replace(minute=0,second=0,microsecond=0)
        self.base_dir = os.path.join(scripts_dir,'text')
        self.raw_dir = os.path.join(self.base_dir,'raw')
        self.processed_dir = os.path.join(self.base_dir,'processed')  
        self.staged_dir = os.path.join(self.base_dir,'staged')
        self.trimmed_file_list= []
        self.colnames = []
        self.totnames = []
        self.df_master = None
        for sf in os.listdir(self.raw_dir):
            os.remove(os.path.join(self.raw_dir,sf))
        for st in os.listdir(self.staged_dir):
            os.remove(os.path.join(self.staged_dir,st))
        self.colsp = 0.4
        self.model_info = {'gfs3':{'cmap':plt.cm.Blues, 'cspace': self.colsp, 'width': 1},
                      'nam':{'cmap':plt.cm.Greens, 'cspace': self.colsp, 'width': 1},
                      'hrrr':{'cmap':plt.cm.Greens, 'cspace': self.colsp, 'width': 1},    
                      'rap':{'cmap':plt.cm.Blues, 'cspace': self.colsp, 'width': 1}}
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
        self.gfs_start = self.roundup6 + timedelta(days=1)
        self.gfs_end = self.roundup6 + timedelta(hours=84)
        #clean = goback.replace(minute=0, second=0, microsecond=0)
        # hourly run time interval for these models, could go > 4 versions back

        if self.model == 'hrrr' or self.model == 'rap':
            
            for h in range (0,6):
                new = goback - timedelta(hours=h)
                hr = datetime.strftime(new, '%H')
                tf = datetime.strftime(new, '%Y%m%d_%H')
                psbl_file = '{}_{}_{}.txt'.format(tf,self.model,self.station)
                self.psbl_files.append(psbl_file)
            self.hrs.append(hr)
            
        elif self.model == 'nam' or self.model == 'namnest' or self.model == 'gfs3':
            # 6 hourly run time interval for these models
            for i in range (0,4):
                new = goback6 - timedelta(hours=i*6)
                hr = datetime.strftime(new, '%H')
                tf = datetime.strftime(new, '%Y%m%d_%H')
                psbl_file = '{}_{}_{}.txt'.format(tf,self.model,self.station)
                self.psbl_files.append(psbl_file)
        else:
            print('model not found!')

        print(self.psbl_files)
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


    def stage_files(self):
        """
        Based on the station and model associated with this object instance, we can
        copy the pertinent files from the processed directory into the staging directory.

        """
        self.model_count = 0
        self.already = sorted(os.listdir(self.processed_dir),reverse = True)
        for f in self.already:
            if self.station in f and self.model in f and self.model_count < 4:
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
            fname = os.path.join(self.staged_dir,f)
            fhrs = None
            self.df = None
            find_dts = re.compile('\d{8}_\d{2}')
            #find_mod_name = re.compile('(?<=_)\S+(?=\.)')
            dts_m = find_dts.search(str(fname))
            dts = dts_m[0]
            cols = ['FH', 'Day' ,'Mon', 'date' ,'hour', 'Wind','SfcT','Ptype','Snow','TotSN','ObsSN','Sleet',
                'TotPL','FZRA','TotZR','QPF','TotQPF','SRat','CumSR','ObsSR','PcntSN','PcntPL','PcntRA','pcpPot','pthick']

            issued = datetime.strptime(dts, '%Y%m%d_%H')
            colname = datetime.strftime(issued, '%m%d%H')
            colname = colname + str(fname)[-7:-4]
            self.df = pd.read_csv(fname, names=cols)
            
            self.df['Wind'] = [int(str(wspd)[-4:-2]) for wspd in self.df['Wind']]
            self.df['SfcT'] = [float(str(Temp)[:-1]) for Temp in self.df['SfcT']]
    
            fhrs = list(self.df.FH.values)
            dts = []
            for fh in range(0,len(fhrs)):
                hrs = int(fhrs[fh])
                dt = issued + timedelta(hours=hrs)
                dts.append(dt)

            self.df['Datetime'] = dts

            try:
                self.df.drop(labels=['FH','Day','Mon','Date','Hour','pthick'],axis=1, inplace=True)
            except:
                pass

            self.df = self.df.set_index(pd.DatetimeIndex(self.df['Datetime']))
            self.df = self.df.rename_axis(None)
            if self.df_master is None:
                self.df_master = self.df
            
            sn = self.df['Snow'].astype(float)            
            self.df_master[str(colname)] = sn
            self.colnames.append(colname)

            totsn = self.df['TotSN'].astype(float)
            totname = 'tot' + self.model + colname
            self.df_master[str(totname)] = totsn
            if totname not in self.totnames:
                self.totnames.append(totname)

        return

    def plot_model(self):
        if 'rap' in self.n:
            model = 'rap'
        elif 'hrrr' in self.n:
            model = 'hrrr'
        else:
            model = self.model
        legend = self.n[3:-3] + '00Z'
        cmap = self.model_info[model]['cmap']
        c = self.model_info[model]['cspace']
        rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 4)))
        plt.plot(self.df_master[self.n], color=cmap(c), linewidth=3)
        self.custom_line = Line2D(self.df_master[self.n], [0], color=cmap(c), lw=c*5)
        self.legend_title_list.append(legend)
        self.custom_line_list.append(self.custom_line)
        self.model_info[model]['cspace'] = self.model_info[model]['cspace'] + (1 - self.colsp)/4
        return


    def final_plot(self):
        
        self.custom_line_list = []
        self.legend_title_list = []

                
        #hours = mdates.HourLocator()
        #myFmt = DateFormatter("%d%h")
        #self.N = len(self.totnames)

        try:
            f1 = self.df_master[self.totnames[-1]] * 4
            f2 = self.df_master[self.totnames[-2]] * 2
            f3 = self.df_master[self.totnames[-3]] * 1
            f4 = self.df_master[self.totnames[-4]] * 0.5
            self.df_master['weighted'] = f1 + f2 + f3 + f4
            self.df_master['weighted'] = self.df_master['weighted']/7.5
        except:
            self.df_master['weighted'] = f1/4            



        for self.n in self.totnames:
            self.plot_model()

        self.weighted_line = Line2D(self.df_master['weighted'], [0], color='r', lw=2)
        self.custom_line_list.append(self.weighted_line)
        self.legend_title_list.append('Weighted Avg\n(last 4 runs)')
        plt.plot(self.df_master['weighted'], color='r',linewidth=2, linestyle='-',)
        #plt.ylim(0, 2*np.max(self.df_master['weighted']) )
        plt.grid(axis='y')
        self.ax.legend(self.custom_line_list, self.legend_title_list)
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)

        title_str = 'Snow Accum\nstation: {}'.format(self.station)
        self.ax.text(0.23, 0.95, title_str, transform=self.ax.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)


        if 'gfs3' in self.model_list:
            self.ax.set_xlim(pd.Timestamp(self.gfs_start), pd.Timestamp(self.gfs_end))
        #self.ax.xaxis.set_major_locator(hours)
        #self.ax.xaxis.set_major_formatter(myFmt)
        img_fname = '{}_{}.png'.format(self.model,self.station)
        image_dst_path = os.path.join(image_dir,img_fname)
        plt.savefig(image_dst_path,format='png')
        plt.close()


                          

        return



#gfk = SnowTool('kgfk','hrrr')
#mqt = SnowTool('kmqt','gfs3')
#tlh = SnowTool('ktlh','hrrr')
#biv = SnowTool('biv','gfs3')
grr = SnowTool('kgrr',['hrrr','rap'])#['gfs3','nam'])
#mbl = SnowTool('kmbl','nam')