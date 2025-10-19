#Autora: Gema Antón

# Archivo EDA para cualquier tipo de dataset
#Trabajará principalmente con csv


# Inicio, importar librerias necesarias:
# pandas: tratamiento de DataFrames
# numpy: control numérido
# matplotlib: visualizaciones
# seaborn: visualizaciones avanzadas


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")

import scipy.stats as st
import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest


def EDA(df):
        
        
        print(f"La forma:")
        print(f"{df.shape}\n")
        print(f"\n-----------------------------\n")
        print(f"Las columnas:")
        print(f"{df.columns}\n")
        print(f"\n-----------------------------\n")
        print(f"Los tipos de datos:")
        print(f"{df.dtypes}\n")
        print(f"\n-----------------------------\n")
        print(f"Los nulos:")
        print(f"{df.isnull().sum()}\n")
        print(f"\n-----------------------------\n")
        print(f"Los duplicados:")
        print(f"{df.duplicated().sum()}\n")
        print(f"\n-----------------------------\n")
        print(f"Los principales estadísticos:")
        print(f"{df.describe().T}\n")

        print(f"Las modas de las columnas categóricas:\n")
        columnas_cat = df.select_dtypes(include = 'object')
        for columna in columnas_cat:
            if df[columna].isnull().any():
                print(f"Revisando {columna}")
                print(df[columna].value_counts())  #mode()[0]
                print("") 
        print(f"\n-----------------------------\n")

        print(" COLUMNAS CATEGÓRICAS")
        col_obj = df.select_dtypes(include='object').columns
        #Lista para guardar columnas

        found_obj = False  #Manera de comprobar que vamos a encontra nulos las columnas categoricas
        #Recorremos el bucle columna por columna
        for col in col_obj:
            
            por = df[col].isnull().mean() * 100
            if por > 0:
                print(f" - {col}: {por:.0f}% de nulos")
                #Si el porcentaje es de X se podria comparar aqui y tratar


                found_obj = True #Check de que ha encontrado nulos

        if not found_obj:
            print(" No hay nulos en columnas categóricas")


        #Ahora comprobación de columas de tipo numérico
        print(" COLUMNAS NUMÉRICAS")
        col_num = df.select_dtypes(include='number').columns
        
        found_num = False
        for col in col_num:
            por = df[col].isnull().mean() * 100
            if por > 0:
                print(f" - {col}: {por:.0f}% de nulos")
                found_num = True
        if not found_num:
            print(" No hay nulos en columnas numéricas")


        return



def tratar_null(df):
    
    col_obj = df.select_dtypes(include='object').columns        
    col_num = df.select_dtypes(include='number').columns
    #Lista para guardar columnas, trataremos una a una las nulas, dependiendo el tipo de dato y % de datos
    #Recorremos el bucle columna por columna
    for col in col_obj:
        por = df[col].isnull().mean() * 100
        if por == 0:
            print(f"No hay nulos en: {col} ")
        else:    
            if por < 20:
                # Pocos nulos: imputar con la moda (valor más frecuente)
                print(f"Tratando columna: {col} ({por:.0f}% nulos) -> Moda")
                df[col] = df[col].fillna(df[col].mode()[0])

            elif 20 <= por < 80:
                # Bastantes nulos: imputar con valor genérico
                print(f"Tratando columna: {col} ({por:.0f}% nulos) -> 'Unknown'")
                df[col] = df[col].fillna('Unknown')

            else:
                # Demasiados nulos: se puede optar por eliminar o marcar aparte
                print(f"Columna {col} tiene {por:.0f}% nulos. Considerar eliminarla o analizarla aparte.")
                df.drop(df[col], axis=1, inplace=True)
                #df[col] = df.drop(df[col], axis=1, inplace=True)
                

            #Si el porcentaje es de X se podria comparar aqui y tratar
            #Ahora comprobación de columas de tipo numérico
    for col in col_num:
        por = df[col].isnull().mean() * 100
        if por == 0:
                print(f"No hay nulos en: {col} ")
        
        else:    
            if por < 20:
                # Pocos nulos: imputar con la media
                print(f"Tratando columna: {col} ({por:.0f}% nulos) -> Media")
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]])

            elif 21 <= por < 70:
                # Porcentaje medio-alto de nulos: imputar con KNN
                print(f"Tratando columna: {col} ({por:.0f}% nulos) -> KNNImputer")
                imputer_knn = KNNImputer(n_neighbors=5)
                df[col] = imputer_knn.fit_transform(df[[col]])

                # Cambio

            elif por > 71:
            # Demasiados nulos: imputar con un valor neutro o eliminar
                print(f"Columna {col} tiene {por:.0f}% nulos. Considerar eliminarla o imputar manualmente.")
                #df=df.drop(df[col], axis=1)
        #Ya tratadito devolvemos el df
    
    return df


        