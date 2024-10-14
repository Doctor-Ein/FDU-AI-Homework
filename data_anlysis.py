# with open("OutputTemp.txt","r") as f:
# 	a=f.readline()
# print(type(a))

# with open("OutputTemp.txt","r") as f:
# 	for i in range(1024):
# 		a = f.readline()

import pandas as pd

# # 使用正则表达式来匹配逗号和分号
# df = pd.read_csv('OutputTemp.txt', sep=r'[:,]', engine='python')
#
# # 导出为 Excel 文件
# df.to_excel('output.xlsx', index=False)
# print("数据已成功导出到 Excel 文件")

# df = pd.read_csv('OutputTemp2.txt', sep=r'[:,]', engine='python')
#
# with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='a') as writer:
#     df.to_excel(writer, sheet_name='Sheet2', index=False)

df = pd.read_csv('OutputTemp3.txt', sep=r'[:,]', engine='python')

with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='a') as writer:
    df.to_excel(writer, sheet_name='Sheet3', index=False)

