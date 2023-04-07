import fileinput
import re
import os
import glob

path = "app/nucc"
file_path_lists = glob.glob("{}/**txt".format(path), recursive=True)
# ファイル作成
new_filename = 'app/nucc/train.txt'
with open(new_filename, 'w'):
    pass

# 編集&書き込み
with open(new_filename, mode='w') as newFile:
    with fileinput.FileInput(files=(file_path_lists)) as f:
        print(f.filename())
        for line in f:
            repStr = re.sub('^＠.+\n', '', line)
            if repStr != '':
                repPrintStr = re.sub('^[A-Z][0-9][0-9][0-9]：', '', line)
                newFile.write(repPrintStr)
