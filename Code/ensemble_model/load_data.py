"""
Base IO code for datasets
data sources: uci  Polish companies bankruptcy data
"""
#author ShuoshuoFan
#Email  shuoshuoFan@gmail.com


from os.path import join
from pandas import read_csv

def load_bankdata(file_name):
    """
    loads data from module_path/file_name

    Parameters
    ----------
    file_name:String.Name of csv file to be load

    Returns
    ---------
    data:   Numpy Array ,features of data

    target: Numpy Arra , lables of data
    """
    module_path='../../Data/csv/'
    bank_data=read_csv(join(module_path,file_name))
    #convert datafram to numpy
    tmp=bank_data.as_matrix()
    data=tmp[:,:-1]
    target=tmp[:,-1]
    return data,target

def load_1year():
    """
    load bankruptcy data ,the first year
    """
    file_name='1year.arff.csv'
    return load_bankdata(file_name)


def load_2year():
    """
    load bankruptcy data ,the second year
    """
    file_name='2year.arff.csv'
    return load_bankdata(file_name)


def load_3year():
    """
    load bankruptcy data ,the third year
    """
    file_name='3year.arff.csv'
    return load_bankdata(file_name)

def load_4year():
    """
    load bankruptcy data ,the first year
    """
    file_name='4year.arff.csv'
    return load_bankdata(file_name)

def load_5year():
    """
    load bankruptcy data ,the first year
    """
    file_name='5year.arff.csv'
    return load_bankdata(file_name)
