import os

AccelTapTrainPath = './data/TestDataFile/Source'
ImageNFTrainPath = './data/TrainDataFile/Target'
#
# for filename in os.listdir(AccelTapTrainPath): # 批量改加速度扫描文件名。有bug，不想改了。先执行第一个if，注释掉后再执行第二个if
#     # if '_0_' in filename:
#     #     AccelOld_0 = os.path.join(AccelTapTrainPath, filename)
#     #     AccelNew_0 = os.path.join(AccelTapTrainPath, filename.split('_')[0] + filename.split('_')[1] + '_' + filename.split('_')[2] + '_' + filename.split('_')[3] + '_' + filename.split('_')[4])
#     #     os.rename(AccelOld_0, AccelNew_0)
#     #     # print(AccelOld_0)
#     #     # print(AccelNew_0)
#
#
#     if 'Z' in filename:
#         AccelOld = os.path.join(AccelTapTrainPath, filename)
#         AccelNew = os.path.join(AccelTapTrainPath, filename.split('_')[0] + '_' + filename.split('_')[3])
#         os.rename(AccelOld, AccelNew)
#         # print(AccelOld)
#         # print(AccelNew)


for filename in os.listdir(ImageNFTrainPath): # 批量改图片扫描文件名。有bug，不想改了。先执行第一个if，注释掉后再执行第二个if
    # if '_0_' in filename:
    #     AccelOld_0 = os.path.join(ImageNFTrainPath, filename)
    #     AccelNew_0 = os.path.join(ImageNFTrainPath, filename.split('_')[0] + filename.split('_')[1] + '_' + filename.split('_')[2] + '_' + filename.split('_')[3] + '_' + filename.split('_')[4])
    #     os.rename(AccelOld_0, AccelNew_0)
    #     # print(AccelOld_0)
    #     # print(AccelNew_0)

    # if 'train0' in filename:
    #     ImageOld_0 = os.path.join(ImageNFTrainPath, filename)
    #     ImageNew_0 = os.path.join(ImageNFTrainPath, filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2] + '_' + 'train10.jpg')
    #     os.rename(ImageOld_0, ImageNew_0)

    if '.jpg' in filename:
        ImageOld = os.path.join(ImageNFTrainPath, filename)
        ImageNew = os.path.join(ImageNFTrainPath, filename.split('_')[0] + '_' + filename.split('_')[3])
        os.rename(ImageOld, ImageNew)



