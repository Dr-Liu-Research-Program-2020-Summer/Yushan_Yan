from Correlation import *



variables = {'a':'AirFlow','dp':'Damper Position','dat':'Discharge Air Temperature','zt':'Zone Temperature','as':'Airflow Setpoint','hwvc':'Hot Water Valve Command '}
var_1 = variables[input('var1:')]
var_2 = variables[input('var2:')]
def test(sample = 20,var_1 = var_1,var_2 = var_2):

    df_row = pd.DataFrame()
    for path in range(0,sample):
        df2 = read_in(var_1 =var_1,var_2= var_2,path = path,time_interval = 120)
        df_row = pd.concat([df_row, df2])
        #print(type(df_row))
    df_row.to_excel(str(sample)+'samples' + 'input Data.xls')   
    return df_row
#test()