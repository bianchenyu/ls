import os
from data_transfer import transfer

def process(dataPath):
    dataList = os.listdir(dataPath)
    for data in dataList:
        path = dataPath + '\\' + data
        folder = os.listdir(path + '\\audio')
        for af in folder:
            # add suffix '.jpg'
            f = af[:-4]
            image_path = path + r'\video' + '\\' + f
            fileList = os.listdir(image_path)
            currentpath = os.getcwd()
            os.chdir(image_path)
            for file in fileList:
                if os.path.splitext(file)[-1] != '.jpg':
                    os.rename(file, file + '.jpg')
            os.chdir(currentpath)

            # process openFace
            os.chdir(r"E:\ls\人脸表情动画\OpenFace\x64\Release")
            cmd = '.\FeatureExtraction.exe ' + ' -fdir ' + path + r'\video' + '\\' + f + ' -out_dir ' + path + r'\processed' + '\\' + f
            print(cmd)
            os.system(cmd)
            os.chdir(currentpath)

            transfer(path + r'\processed' + '\\' + f, f)




datapath = r"E:\ls\人脸表情动画\数据集\vidtimit"
# folder = ['sa1', 'sa2', 'si649', 'si1279', 'si1909', 'sx19', 'sx109', 'sx199', 'sx289', 'sx379']
process(datapath)
# for f in folder:
#     path = datapath + r'\processed' + '\\' + f
#     transfer(path ,f)