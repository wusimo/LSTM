import pandas as pd
from datetime import datetime
import math

tag = 'value'
file_path = './Desktop/'

csv_file_name = '6.csv'

time_zone = 'Europe/London'
# set time zone
local_tz = pytz.timezone(time_zone)

# read energy load from csv file
energy_load_read = pd.DataFrame.from_csv(file_path+csv_file_name)[tag]

# convert energy load's index from timestamp to datetime format
energy_index = [datetime.fromtimestamp(x,tz=local_tz) for x in energy_load_read.index]

# reset the load's index
energy_load = pd.DataFrame({'Value':list(energy_load_read)},index=energy_index).resample('H')


def filter_num( x):
		try:
			return np.float(x)
		except:
			return np.nan

# import hourly temp
file_path = './Desktop/'

file_name = 'temp_hourly.csv'

temp = pd.DataFrame.from_csv(file_path+file_name)
temp = temp[['WetBulbFarenheit','Date','Time']]

# store temp in data, datetime in index
data = [value for value in temp['WetBulbFarenheit']]
index = []


for j in temp.iterrows():
    c = datetime(int(str(j[1].Date)[0:4]),int(str(j[1].Date)[4:6]),int(str(j[1].Date)[6:8]),int(math.floor(j[1].Time/100)),int(j[1].Time-(math.floor(j[1].Time/100)*100)))
    index.append(c)
ts_temp = pd.DataFrame({'temp':data},index = index)
ts_temp['temp'] = ts_temp['temp'].apply(filter_num)

hourly_temp = ts_temp.resample('H').fillna(method='ffill')
temp_series = hourly_temp.copy()

n_steps = 6
n_inputs = 4
length =  min(len(energy_load),len(temp_series))
X_train = np.empty((length,n_steps,n_inputs))

for i in range(1,length-n_steps):
    for j in range(1,n_steps):
        X_train[i-1,j-1,0] = temp_series.temp[i-1+j-1]
        X_train[i-1,j-1,1] = energy_load.values[i-1+j-1]
        X_train[i-1,j-1,2] = energy_load.index[i-1+j-1].time().hour
        X_train[i-1,j-1,3] = energy_load.index[i-1+j-1].isoweekday()

y_train = np.empty((length,1))
for i in range(1,length-n_steps):
    y_train[i,0] = energy_load.values[i-1+n_steps]