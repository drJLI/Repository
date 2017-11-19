# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import datetime,os
import eikon as ek
from dateutil.parser import parse

def years_ago(end_date, years):
    if type(end_date) == str:
        end_date = parse(end_date)
    if end_date.month == 2 and end_date.day == 29:
        end_date = end_date.replace(day=28)
    return end_date.replace(year=end_date.year - years).replace(month = end_date.month - 1)

len_hist = 5
end_date = datetime.date.today()
dates = pd.bdate_range(years_ago(end_date,len_hist),end_date)
df_TS = pd.DataFrame(index=dates).sort_index(axis = 0,ascending = False)

cpi = ['AAPL.O']
pwd = os.getcwd()
#ek.set_app_id('8B2D38D084ECE14EBF190BC7')
ek.set_app_id('B2AA4B3A4A8EE1BCEFA9B')

df2,err = ek.get_data(cpi,['TR.PriceClose.date','TR.PriceClose','TR.PriceClose.currency','TR.EV',
                           'TR.PE','TR.CompanyMarketCap','TR.WACCBeta','TR.WACCCostofEquity',
                           'TR.TotalReturn1Wk','TR.Volume'],
                     {'SDate': 0, 'EDate': -len_hist*252, 'FRQ': 'D'})

df1,err = ek.get_data(cpi,['TR.NormalizedEbitda.date','TR.NormalizedEbitda','TR.TotalEquity','TR.TotalCurrentAssets','TR.NetIncome'],{'SDate': 0, 'EDate': -len_hist*4, 'FRQ': 'FY'})

df_bm,err = ek.get_data('.SPLRCTHSP',['TR.PriceClose.date','TR.PriceClose'],{'SDate': 0, 'EDate': -len_hist*252, 'FRQ': 'D'})

df_spx,err = ek.get_data('.SPX',['TR.PriceClose.date','TR.PriceClose'],{'SDate': 0, 'EDate': -len_hist*252, 'FRQ': 'D'})
df2.set_index(['Date'],inplace = True)
#df1.set_index(['Date'],inplace = True)
df_spx.set_index(['Date'],inplace = True)
df_bm.set_index(['Date'],inplace = True)
#df1 = df1.drop('Instrument',axis = 1)

df_bm = df_bm.rename(columns={'Price Close': 'SPLRCTHSP'} ).drop('Instrument',axis = 1)
df_spx = df_spx.rename(columns={'Price Close': 'SPX'}).drop('Instrument',axis = 1)
df = df2.join(df_spx,how = 'left').join(df_bm,how = 'left') #.join(df1,how = 'left')

df.to_csv(os.path.join(pwd,'data.csv'))
