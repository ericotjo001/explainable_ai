import pandas as pd

print('expected total = ', 4*256* (9*56+ 1) ) # 517120

df = pd.read_csv('mabmodel_10_1_metric_aggreagate.csv', )
print(len(df))
# print(xai_method, len(df[df['xai_method']==xai_method]))

# for k,xai_method in enumerate([
#     'Saliency',
#     'IntegratedGradients', 
#     'InputXGradient', 
#     'DeepLift', 
#     'GuidedBackprop', 
#     'GuidedGradCam', 
#     'Deconvolution',
#     'GradientShap',
#     'DeepLiftShap',
#     'mab',
# ]): 
# 	print('\n\n',xai_method)
# 	for i in range(4):
# 		lengs = []
# 		for j in range(256):
# 			df1 = df[df['id'] == f'data-3_{i}.shard_{i}']
# 			df2 = df1[df1['xai_method'] == xai_method]
# 			lengs.append(len(df2))
# 		print('shard:',i, lengs)
