from Correlation import *

lag_limit = 20
table = pd.DataFrame(columns = ['AirFlow','Damper Position','Discharge Air Temperature','Zone Temperature','Airflow Setpoint','Hot Water Valve Command '])
faults,list_rooms = read_in(start=0,end=25,method = 'pearson' ,var_1='AirFlow',var_2='Damper Position')

table['rooms'] = list_rooms
table = table.set_index('rooms')
table.loc[:,:] = 0
print(table)
variables = {'a':'AirFlow','dp':'Damper Position','dat':'Discharge Air Temperature','zt':'Zone Temperature','as':'Airflow Setpoint','hwvc':'Hot Water Valve Command '}
#var_1 = variables[input('variable one:')]
#var_2 = variables[input('variable two:')]

def make_table(var_1='AirFlow',var_2='Damper Position',table = table,method = 'pearson',limit = 1730,time_interval = 120):
    
    start = 0 
    end = 0 
    while start < limit and end < limit:
        end = start + int(time_interval/5) 
        faults,list_rooms = read_in(start=start,end=end,method = method ,var_1=var_1,var_2=var_2)
        
        for i in faults.index.tolist():
            table.loc[i,var_1] += 1
            table.loc[i,var_2] += 1

        start = end
    return table
table = make_table(var_1='AirFlow',var_2='Airflow Setpoint',table = table,method = 'pearson')
table = make_table(var_1='Hot Water Valve Command ',var_2='Discharge Air Temperature',table = table,method = 'pearson')
table = make_table(var_1='Discharge Air Temperature',var_2='Zone Temperature',table = table,method = 'spearman')
table = make_table(var_1='AirFlow',var_2='Damper Position',table = table,method = 'pearson')
table.to_excel('Correlation_table.xls')