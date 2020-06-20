import csv
import os

def transfer(path, name):
    #name = '2019-11-28-18-30'

    currentpath = os.getcwd()
    os.chdir(path)
    # creat new csv format
    f = open(name + '_new.csv', 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['frame', ' timestamp', ' confidence', ' success', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
         ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
         ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c',
         ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c'])

    with open(name + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_writer.writerow(
                [row['frame'], row[' timestamp'], row[' confidence'], row[' success'], row[' AU01_r'], row[' AU02_r'],
                 row[' AU04_r'], row[' AU05_r'], row[' AU06_r'], row[' AU07_r'], row[' AU09_r'], row[' AU10_r'],
                 row[' AU12_r'], row[' AU14_r'], row[' AU15_r'], row[' AU17_r'], row[' AU20_r'], row[' AU23_r'],
                 row[' AU25_r'], row[' AU26_r'], row[' AU45_r'], row[' AU01_c'], row[' AU02_c'], row[' AU04_c'],
                 row[' AU05_c'], row[' AU06_c'], row[' AU07_c'], row[' AU09_c'], row[' AU10_c'], row[' AU12_c'],
                 row[' AU14_c'], row[' AU15_c'], row[' AU17_c'], row[' AU20_c'], row[' AU23_c'], row[' AU25_c'],
                 row[' AU26_c'], row[' AU28_c'], row[' AU45_c']])

    f.close()
    os.chdir(currentpath)

