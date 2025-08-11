#
# Project: ZULS_0004
# Author : Marcel Beuler
# Date   : 2025-07-25
#

import os
import numpy as np
import csv

##########################################################################################
def readCSVData():
    # Get a list of all csv files stored in current working directory (Data)
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]

    # Count the number of csv files
    number_csv_files = len(csv_files)

    # Get a list of all valid csv files by checking the header
    # 9 columns: SystemTime, AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ,
    #            RollAngleNF, PitchAngleNF
    valid_csv_files = []
    stored_data = []
    for f in csv_files:
        csv_file_path = os.getcwd() + '\\' + f
        with open(csv_file_path, 'r') as file:
            read_obj = csv.reader(file)
            header = next(read_obj)
            columns = len(header)

            if columns == 9:
                if header[0] == 'SystemTime' and \
                   header[1] == 'AccelX' and header[2] == 'AccelY' and \
                   header[3] == 'AccelZ' and \
                   header[4] == 'GyroX' and header[5] == 'GyroY' and \
                   header[6] == 'GyroZ' and \
                   header[7] == 'RollAngleNF' and header[8] == 'PitchAngleNF':
                    valid_csv_files.append(f)
                    stored_data.append('--> Sensor Values & Euler Angles')

    # Print number of valid csv files, their names and kind of stored data
    # (Raw Values)
    number_valid_csv_files = len(valid_csv_files)
    if number_valid_csv_files <= 1:
        print(f'There is {number_valid_csv_files} valid csv file in the current folder:')
    else:
        print(f'There are {number_valid_csv_files} valid csv files in the current folder:')
    for i in range(number_valid_csv_files):
        print(f"{i+1:2}. {valid_csv_files[i]} {stored_data[i]}")

    # User can select one file
    if number_valid_csv_files > 1:
        while True:
            keyboard = input('Please enter the number of the disired file (cancel with n): ')
            try:
                selector = int(keyboard)
                if 1 <= selector <= number_valid_csv_files:
                    selected_file = valid_csv_files[selector - 1]
                    print(f'You have selected the file "{selected_file}".')
                    break
                else:
                    print('Invalid selection!')
            except ValueError:
                if keyboard == 'N' or keyboard == 'n':
                    print('')
                    return np.array([])
    elif number_valid_csv_files == 1:
        selected_file = valid_csv_files[1 - 1]
    else:
        print('No csv file selectable.')
        print('')
        return np.array([])

    csv_file_path = os.getcwd() + '\\' + selected_file
    with open(csv_file_path, 'r') as file:
        read_obj = csv.reader(file)

        # Read header
        header = next(read_obj)

        # Number of columns and rows
        columns = len(header)
        #print('columns =', columns)

        rows = 0
        for i in read_obj:
            rows += 1
        #print('rows    =', rows)

    # Columns (uint32, int16, int16, int16, int16, int16, int16, float32, float32)
    # We use a structured array whose datatype is a composition of simpler datatypes
    dtype_t = [('col1', np.uint32), \
               ('col2', np.int16),  \
               ('col3', np.int16),  \
               ('col4', np.int16),  \
               ('col5', np.int16),  \
               ('col6', np.int16),  \
               ('col7', np.int16),  \
               ('col8', np.float32),\
               ('col9', np.float32)]     # Specify data type for each column
    Data = np.zeros(rows, dtype=dtype_t) # type is class 'numpy.ndarray'

    with open(csv_file_path, 'r') as file:
        read_obj = csv.reader(file)

        # Read header
        header = next(read_obj)

        for m in range(0, rows):
            Dataline = next(read_obj)
            Data[m][0] = np.uint32(Dataline[0])  # SystemTime
            Data[m][1] = np.int16(Dataline[1])   # AccelX
            Data[m][2] = np.int16(Dataline[2])   # AccelY
            Data[m][3] = np.int16(Dataline[3])   # AccelZ
            Data[m][4] = np.int16(Dataline[4])   # GyroX
            Data[m][5] = np.int16(Dataline[5])   # GyroY
            Data[m][6] = np.int16(Dataline[6])   # GyroZ
            Data[m][7] = np.float32(Dataline[7]) # RollAngleNF
            Data[m][8] = np.float32(Dataline[8]) # PitchAngleNF

    return Data
##########################################################################################

