import sql_load as sql
import pandas as pd



df1 = pd.read_csv('IPV_cleaned.csv')
df2 = pd.read_csv('renta_cleaned.csv')
df3 = pd.read_csv('spain_realState.csv', sep=';')



sql.connect_to_mysql()
sql.load_data_to_mysql(df1,df2,df3)
#sql.drop_table()