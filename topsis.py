# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:25:45 2020

@author: ayush
"""

"""
Created on Wed Nov 11 17:32:18 2020
@author: ayush
"""
import pandas as pd
import copy 
import numpy as np
import sys
weights = []
#***********************************************************************************************************
def getInput(input_data,argv):
    global weights
    #check output file extension
    if argv[4][-4:]!=".csv" and argv[4][-4:]!=".txt":
        print("Output file not a valid file!! Must be a .csv or .txt file")
        sys.exit()
    #Checking file has atleast three columns!
    if input_data.shape[1]-1 < 3:
        print("Given file does not have atleast three or more columns!!")
        sys.exit()
    #Check the type of columns 2 to end is numeric only
    for i in range(1,input_data.shape[1]):
        if not np.issubdtype(input_data.iloc[:,i],np.number):
            print("Column number ",i," is not numeric form!")
            sys.exit()
    #Obtaining the weights array
    try:
        StringWeights = list(argv[2].split(","))
        weights = []
        for weight in StringWeights:
            weights.append(float(weight))
    except:
        #Check if all weights were not numeric!
        print("All weights were not numeric! Or they are not seperated by commas!")
        sys.exit()
    if len(weights) < input_data.shape[1]-1:
        #Check if all weights were not seperated by commas or they were not the right number!
        print("Not accurate number of weights given! They should be equal to the number of the features!")
        sys.exit()
    #Obtaining the impacts array:
    impactsString = list(argv[3].split(","))
    #If inccorect number of impacts given or not seperated by commas
    if len(impactsString) < input_data.shape[1]-1:
        print("Incorrect number of impacts! Or they were not seperated by commas!")
        sys.exit()
    for i in range(0,len(impactsString)):
        #Given impact was not '+' or '-'
        if impactsString[i] not in ('+','-'):
            print("Incorrect impact given. It must be '+' or '-'.")
            sys.exit()
        #Other wise convert the weight negative when there is a '-' else leave it as it is
        if impactsString[i] == '-':
            weights[i] = -1 * weights[i]
#************************************************************************************************
def solve(input_data,argv):
    global weights
    output_data = copy.deepcopy(input_data)
    #Normalizing the data in every column
    for cols in input_data.iloc[:,1:]:
        sum = 0
        for values in input_data[cols]:
            sum = sum + values**2
        input_data[cols] = input_data[cols]/np.sqrt(sum)
    #Multiplying each column with its weight
    i = 0
    for col in input_data.iloc[:,1:]:
        input_data[col] = input_data[col]*weights[i]
        i+=1
    #Calculate the ideal best and ideal worst for each column
    Vplus = []#Ideal Best matrix
    Vminus = []#Ideal Worst matrix
    for cols in input_data.iloc[:,1:]:
        Vplus.append(max(input_data[cols]))
        Vminus.append(min(input_data[cols]))
    #Find the euclidean distance from ideal best and ideal worst value!
    Splus = []
    Sminus = []
    for rows in range(input_data.shape[0]):
        Sijplus = 0
        Sijminus = 0
        i = 0
        for cols in input_data.iloc[:,1:]:
            Sijplus += (input_data.iloc[rows][cols] - Vplus[i])**2
            Sijminus += (input_data.iloc[rows][cols] - Vminus[i])**2
            i+=1
        Splus.append(np.sqrt(Sijplus))
        Sminus.append(np.sqrt(Sijminus))
    Ssum = []
    Performance = []
    for i in range(len(Splus)):
        sum = Splus[i] + Sminus[i]
        Ssum.append(sum)
        Performance.append(Sminus[i]/sum)
    #Making the final output dataframe with two extra columns
    output_data["TOPSIS"] = Performance
    output_data["Rank"] = output_data["TOPSIS"].rank(ascending = False)
    output_data.to_csv(argv[4])
#******************************************************************************************
#MAIN PROGRAM
#path = 'C:\\Users\\ayush\\Desktop\\5th Semester\\Data Science Fundamentals\\Lab Assignments\\Assignment 6\\Input files for Assignment06'
def topsis(input_file_name,W,I,output_file_name):
    argv = [None,input_file_name, W, I, output_file_name]
    #Incorrect number of arguements
    if len(argv) < 5:
        print("Not accurate arguments!!")
        print("Example: python topsis.py <InputDataFIle> <Weights> <Impacts> <ResultFileName>")
        sys.exit()
    #input the csv file
    try:
        datafile = argv[1]
        if datafile[-4:]!=".csv" and datafile[-4:]!=".txt":
            print("Input file not a valid file!! Must be a .csv or .txt file")
            sys.exit()
        input_data = pd.read_csv(datafile)
    except: 
        #File Not Found
        print("File with given name does not exists! Please check the file name")
        sys.exit()
    getInput(input_data,argv)
    solve(input_data,argv)
    print("Excecuted successfully!!")
