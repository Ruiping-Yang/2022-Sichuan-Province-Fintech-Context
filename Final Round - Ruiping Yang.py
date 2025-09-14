#!/usr/bin/env python
# coding: utf-8

# # Step 1: 程序包/数据准备

# In[1]:


# 导入程序包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import toad
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,auc
from multiprocessing import cpu_count
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# 一般化设置
plt.rcParams['font.sans-serif']=['SimHei'] #配置字体，显示中文
plt.rcParams['axes.unicode_minus']=False #配置坐标轴刻度值模式，显示负号
plt.rcParams['figure.dpi']=100 #设置图片分辨率


# In[3]:


# 导入初赛训练集；复赛训练集、测试集、数据字典
X_train0=pd.read_excel('复赛_trainX.xlsx')
X_train1=pd.read_excel('初赛_trainX.xlsx')
Y_train0=pd.read_excel('复赛_trainY.xlsx')
Y_train1=pd.read_excel('初赛_trainY.xlsx')
X_test=pd.read_excel('testX.xlsx')
dict_df=pd.read_excel('数据字典.xlsx')


# In[4]:


X_train0.head() # 查看复赛训练集


# In[5]:


X_train1.head() # 查看初赛训练集


# In[6]:


X_test # 查看测试集


# In[7]:


Y_train0.head() # 查看复赛标签前5行


# In[8]:


Y_train1.head() # 查看初赛标签前5行


# In[9]:


dict_df.head() # 查看字典前5行


# In[18]:


dict_df.shape # 查看字典大小


# In[11]:


#用于测试集提交的id列单列出来
output =pd.DataFrame()
output['id']=X_test['id']

# 去除复赛数据的id列，用于合并数据集
X_train0.drop(columns='id',inplace=True)
Y_train0.drop(columns='id',inplace=True)
X_test.drop(columns='id',inplace=True)


# In[12]:


# 合并初赛、复赛训练集、标签
X_train=pd.concat([X_train0,X_train1])
Y_train=pd.concat([Y_train0,Y_train1])

# 重新设置行索引，方便后续数据处理
X_train.reset_index(inplace=True,drop=True)
Y_train.reset_index(inplace=True,drop=True)
print(X_train0.shape)
print(X_train1.shape)
print(X_train.shape)
print('--------------')
print(Y_train0.shape)
print(Y_train1.shape)
print(Y_train.shape)


# In[13]:


X_train # 查看训练集合并情况


# In[14]:


Y_train # 查看标签合并情况


# In[15]:


# 将数据字典的解释设置为标签，提高可读性，便于后续特征筛选
dict_ = dict(zip(dict_df['col_name'],dict_df['comment'])) 

# 转换标签
X_train.rename(columns=dict_,inplace=True) 
X_test.rename(columns=dict_,inplace=True)

X_train.head() # 查看复合训练集标签转换情况


# In[16]:


X_test.head() # 查看测试集转换情况


# In[20]:


# 数据预处理准备：合并复合训练集和测试集

# 设定标签，便于后续区分
X_train['train_or_test']=0
X_test['train_or_test']=1

#合并复合训练集和复赛测试集，便于数据处理
data = pd.concat([X_train,X_test],ignore_index=True) 
data.shape


# In[40]:


data.head() 


# # Step 2: EDA探索性分析

# In[22]:


data.describe() # 复合训练集描述性统计


# In[27]:


# 查看标签分布情况

plt.figure(figsize=(4,3)) #创建画布
plt.hist(Y_train['jieju_dubil_status_desc'],bins=10,color='purple')
plt.show()


# In[29]:


# 对比训练集和测试集数据的分布情况
get_ipython().run_line_magic('matplotlib', 'inline')
con_col= list(data.select_dtypes(exclude=object))
obj_col= list(data.select_dtypes(include=object))
cols = 6
rows = len(con_col)
plt.figure(figsize=(4*cols,4*rows))
i=1
for col in con_col:
    ax = plt.subplot(rows,cols,i)
    ax = sns.kdeplot(X_train[col], color="firebrick", fill=True)
    ax = sns.kdeplot(X_test[col], color="navy", fill=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train","test"])
    i+=1
plt.show() # 可以看到，二者分布大体一致


# In[30]:


# 查看变量间相关系数图（选取nunique>200的特征）
fig=[]
for x in con_col:
    if data[x].nunique()>=200:
        fig.append(x)
plt.figure(figsize=(12,12))
f , ax = plt.subplots(figsize = (12,10))
plt.title('Correlation among Constant Variables')
sns.heatmap(data[fig[0:13]].corr(),annot=True,square = True,fmt='.2g',cmap='PuRd')


# In[51]:


# 查看原始数据中高相关性变量的描述性统计情况
high_corr0 = [i for i in con_col if abs(X_train[i].corr(Y_train['jieju_dubil_status_desc']))>0.1][:25]
des_corr0 = data[high_corr0].describe()
des_corr0.plot()
plt.legend(high_corr0,loc='upper left')
plt.title('高相关性变量描述性统计')
plt.show()


# In[97]:


# 查看原始数据中高相关性变量的描述性统计情况
high_corr1 = [i for i in con_col if abs(X_train[i].corr(Y_train['jieju_dubil_status_desc']))<0.01][5:15]
des_corr1=data[high_corr1].describe()
des_corr1.plot()
plt.legend(high_corr1,loc='upper left')
plt.title('低相关性变量描述性统计')
plt.show()


# In[45]:


# 查看原始数据中高iv值变量的描述性统计情况
high_iv0=toad.quality(X_train,Y_train['jieju_dubil_status_desc'],iv_only=True)[:25]
des_iv0=data[high_iv0.index].describe()
des_iv0.plot()
plt.legend(high_iv0.index,loc="upper left")
plt.title('高IV值变量描述性统计')
plt.show()


# In[99]:


# 绘制省份词云图
from string import punctuation
from PIL import Image     
from wordcloud import WordCloud,ImageColorGenerator
import jieba

data0=data.copy()

# 将省份代码转化为省份名称
p={'110000':'北京','120000':'天津','130000':'河北','140000':'山西','150000':'内蒙古','210000':'辽宁','220000':'吉林','230000':'黑龙江','310000':'上海','320000':'江苏','330000':'浙江','340000':'安徽','350000':'福建','360000':'江西','370000':'山东','410000':'河南','420000':'湖北','430000':'湖南','440000':'广东','450000':'广西','460000':'海南','500000':'重庆','510000':'四川','520000':'贵州','530000':'云南','540000':'西藏','610000':'陕西','620000':'甘肃','630000':'青海','640000':'宁夏','650000':'新疆','710000':'台湾','810000':'香港','820000':'澳门'}
for i in  range(len(data0['(客户)证件号码所属省份代码'])):
    j=data0.iloc[i]['(客户)证件号码所属省份代码']
    data0['(客户)证件号码所属省份代码'][i] = p[f'{j}']

#  将院校列转换成列表 便于jieba分词
df_a2=data0['(客户)证件号码所属省份代码'].astype(str).tolist()
add_punc='，。_ '
add_punc=add_punc+punctuation

j=[]
for i in df_a2:
    words = jieba.cut(i,cut_all=True)
    for s in words:
        if s.strip() in add_punc:
            pass
        else:
            j.append(s.strip())

background_Image = np.array(Image.open('PYTHON.jpg'))
img_colors = ImageColorGenerator(background_Image)

plt.figure(figsize=(4,3),dpi=1000)      
mask0= plt.imread('PYTHON.jpg')          

# 制作云图
wc = WordCloud(
    mask=mask0,
    font_path='simhei.ttf',     
    background_color='white',   
    width=1000,
    height=600,
    max_font_size=200,            
    min_font_size=10,
    max_words=50,
    collocations=False,
    font_step=1
)

s=wc.generate(' '.join(j)) # 传入带有空格的数据辅助云图显示
plt.imshow(s,cmap='plasma') # 辅助云图正常显示
plt.axis('off') # 去除横纵坐标轴
plt.show()


# # Step 3: 数据清洗

# In[100]:


df0 = data.copy()


# In[101]:


# 快速浏览数据集
df0.info(verbose=True)


# In[102]:


# 缺失值处理

# 检查各变量是否存在缺失值
df0nu = df0.isnull().sum() # 分别计算每个变量的缺失值个数
print(f'缺失值个数统计：{df0nu.sum()}') # 全部缺失值个数统计
df_nu=df0nu.sort_values(ascending=False) #  按缺失值个数排序
df_nu[0:50].plot.bar(figsize=(20,6),color='sienna')# 绘制缺失值分布图
plt.show()
df_nu.head()


# In[103]:


#删除缺失比例超过90%的特征（根据样本量，特征应至少含有3034个值）
df1 = df0.dropna( axis=1,how='all',thresh=3034) 

#再次查看缺失值
df1nu = df1.isnull().sum()
print(f'删除大量缺失标签后的缺失值个数统计：{df1nu.sum()}')
df1nu.sort_values(ascending=False)


# In[104]:


print(f'df1 shape:{df1.shape}')
df1.info()


# In[105]:


# 填充缺失值
con_col= list(df1.select_dtypes(exclude = object))
obj_col= list(df1.select_dtypes(include=object))
for i in con_col:
    df1[i] = df1[i].fillna(df1[i].median()) # 数值型变量，用中位数填充
for i in obj_col:
    df1[i] = df1[i].fillna(df1[i].mode()[0]) # 类别型变量，用众数填充
df1.isnull().sum().sum() # 检查缺失值是否已全部被填充


# In[106]:


print(f'df1 shape:{df1.shape}')
df1.info()


# In[107]:


# 单一值处理

toad_result=pd.DataFrame(toad.detect(df1))
toad_result


# In[108]:


# 删除unique=1的单一值变量
uni=[]
count=0
for i in toad_result.index:
    if toad_result.loc[i,'unique']==1:
        uni.append(i)
        count=count+1
print(f'单一值特征个数：{count}')
uni


# In[109]:


df2=df1.drop(columns=uni,inplace=False)
df2.shape


# In[110]:


# 检查是否还存在单一值
toad_result=pd.DataFrame(toad.detect(df2))

# 删除unique=1的单一值变量
uni_check=[]
count=0
for i in toad_result.index:
    if toad_result.loc[i,'unique']==1:
        uni_check.append(i)
        count=count+1
print(f'单一值特征个数：{count}')
uni_check


# # Step 4: 数据预处理

# In[111]:


# 处理与日期有关特征的格式

# 提取包含日期的所有列 单独处理
date_df0=df2.filter(like='日期', axis=1)
date_df0.head()


# In[112]:


df3=df2.drop(columns=[x for x in date_df0.columns], inplace=False)
print(f'df2 shape:{df2.shape}')
print(f'df3 shape:{df3.shape}')
print(f'date_df shape:{date_df0.shape}')


# In[113]:


date_df0.describe()


# In[114]:


# 根据描述性统计 查看高度相似的列
import operator
a=operator.eq(date_df0['(借据)展期起始日期'],date_df0['(借据)展期到期日期'])
a_count=0
for i in a:
    if i==False:
        a_count+=1
a_count


# In[115]:


date_df0.dtypes


# In[116]:


# 查看含有0001-01-01的变量
count=0
for i in date_df0.columns:
    if '0001-01-01' in date_df0[i].values:
        print(date_df0[i].value_counts(normalize=True))


# In[117]:


# 对0001-01-01占比高于99%的变量予以删除
date_df1=date_df0.drop(columns=['(借据)展期起始日期', '(借据)展期到期日期','(借据)核销日期'],inplace=False)
date_df1.describe()


# In[118]:


# 对下次重定价日期做独热编码
d=pd.get_dummies(date_df1['(借据)下次重定价日期'],prefix='(借据)下次重定价日期_')
date_df2=date_df1.join(d)
date_df2.drop(columns='(借据)下次重定价日期',inplace=True)
date_df2.head()


# In[119]:


# 对剩下的列使用计数编码
import category_encoders as ce
c=pd.DataFrame()
for i in date_df2.columns[0:6]:
    c[i+'_count']=date_df2[i]
date_df3_= ce.CountEncoder().fit_transform(c)
date_df3_.head()


# In[120]:


date_df3=date_df2.join(date_df3_)
date_df3.head()


# In[121]:


date_df4=date_df3.copy()
for i in date_df4.columns:
    if date_df4[i].dtype=='object':
        date_df4[i]=pd.to_datetime(date_df4[i],format='%Y-%m-%d',errors='coerce')
        date_df4[i] = pd.DatetimeIndex(date_df4[i]).year 
date_df4.head()


# In[122]:


date_df4.info()


# In[123]:


toad.detect(date_df4)


# In[124]:


# 检查各变量是否存在缺失值
nu = date_df4.isnull().sum() # 分别计算每个变量的缺失值个数
print(f'缺失值个数统计：{nu.sum()}') # 全部缺失值个数统计
nu_=nu.sort_values(ascending=False) #  按缺失值个数排序
nu_[0:50].plot.bar(figsize=(20,6),color='sienna')
plt.show()
nu_.head()


# In[125]:


for i in date_df4.columns:
    date_df4[i] = date_df4[i].fillna(0) # 用0填充
nu = date_df4.isnull().sum()
print(f'缺失值个数统计：{nu.sum()}')


# In[126]:


date_df4.info()


# In[127]:


print(df3.shape)
print(date_df4.shape)


# In[128]:


# 将日期变量与主数据集合并
df3=df3.join(date_df4)
df3.head()


# In[129]:


df3.info()


# In[130]:


df3.isnull().sum().sum()


# In[131]:


# 提取包含时间的所有列 单独处理
time_df=df3.filter(like='时间', axis=1)
time_df.head()


# In[132]:


df3.drop(columns='(客户)开立客户时间',inplace=True)


# In[133]:


import time
time_df['(客户)开立客户时间']=pd.to_datetime(time_df['(客户)开立客户时间'])
time_df['开户时长'] = pd.datetime.now() - pd.to_datetime(time_df['(客户)开立客户时间']) # 创建一列差值
time_df['开户时长'] = time_df['开户时长']/pd.Timedelta('1 D')  # 时间差转化为天数
time_df['(客户)开立客户时间']=pd.DatetimeIndex(time_df['(客户)开立客户时间']).year 
time_df.head()


# In[134]:


time_df.info()


# In[135]:


# 将时间变量与主数据集合并
df3=df3.join(time_df)
df3.info()


# In[136]:


# 处理文本型变量

df4=df3.copy()
df4.shape


# In[137]:


obj_col0=list(df4.select_dtypes(include=object))
obj_col0


# In[138]:


obj_df=df4[obj_col0]
obj_df.describe()


# df4.drop(columns=[x for x in obj_df],inplace=True )
# df4.shape

# In[139]:


# 分别查看类别多和类别少的变量
con_obj=[]
count=0
for i in obj_col0:
    if obj_df[i].nunique()>10:
        con_obj.append(i)
        count+=1
print(f'nunique>10的文本型变量个数：{count}')
print(con_obj)

print('------------------------------------------------------------')

dis_obj=[]
count=0
for i in obj_col0:
    if obj_df[i].nunique()<10:
        dis_obj.append(i)
        count+=1
print(f'nunique<10的文本型变量个数：{count}')
print(dis_obj)


# In[140]:


# 查看各标签unique
for i in obj_df.columns:
    print(i)
    print(obj_df[i].unique())
    print('---------------------------')


# In[141]:


# 查看可能相同的列
import operator
a=operator.eq(obj_df['(借据)五级分类代码'],obj_df['(借据)监管五级分类代码'])
a_count=0
for i in a:
    if i==False:
        a_count+=1
a_count


# In[142]:


# 删除其中一个特征
obj_df.drop(columns='(借据)监管五级分类代码',inplace=True)
df4.drop(columns='(借据)监管五级分类代码',inplace=True)
obj_col0.remove('(借据)监管五级分类代码')
dis_obj.remove('(借据)监管五级分类代码')
obj_df.describe()


# In[143]:


# 提取文本型变量中的训练集 准备woe编码
X_train0=df3[df3['train_or_test']==0]
X_train0.head()


# In[144]:


X_train0.shape


# In[145]:


Y_train.shape


# In[146]:


# 查看标签是否对应
for i in dis_obj:
    print(X_train0[i].unique())
    print(obj_df[i].unique())


# In[147]:


obj_df.shape


# In[148]:


#  woe编码：种类较少的文本型变量

for i in dis_obj:
    w=pd.crosstab(X_train0[i],Y_train['jieju_dubil_status_desc'],normalize='columns')
    # 计算woe
    w['woe']=np.log((w.iloc[:,1]+10e-5)/(w.iloc[:,0]+10e-5))
    w['iv']=np.sum(w['woe']*(w.iloc[:,1]-w.iloc[:,0]))
    
    # 替换原始变量
    obj_df[i+'_woe'] = obj_df[i].replace(sorted(obj_df[i].unique()),w['woe'], inplace=False)
obj_df.head()


# In[149]:


# 二值化unique=2的标签
obj_df['(借据)商户编号'].replace(['default','HBD0000001'],[0,1], inplace=True)
obj_df['(借据)数据源'].replace(['M_AG_RETAIL_LOAN_ACCT_H','M_AG_RETL_LOAN_NON_UNITE_ACCT_H'],[0,1], inplace=True)
obj_df['(客户)学业状态'].replace(['结业','毕业'],[0,1], inplace=True)
obj_df.head()


# In[150]:


obj_df.isnull().sum().sum()


# In[151]:


# 标签编码：种类较少的文本型变量
from sklearn.preprocessing import LabelEncoder
for i in dis_obj:
    obj_df[i+'_label'] = LabelEncoder().fit_transform(obj_df[i]) 
obj_df.head()


# In[152]:


obj_df.shape


# In[153]:


# 计数编码：种类较多的文本型变量
for i in con_obj:
    obj_df[i] = ce.CountEncoder().fit_transform(obj_df[i])
obj_df.head()


# In[154]:


obj_df.shape


# In[155]:


df4.shape # 查看数据集大小，确保后续合并无误


# In[156]:


obj_df.drop(columns=dis_obj,inplace=True) 
obj_df.shape


# In[157]:


df4.drop(columns=con_obj,inplace=True) 
df4.shape


# In[158]:


# 将处理后的文本型数据与原数据集合并
df4=df4.join(obj_df)
df4.shape


# In[159]:


df4.isnull().sum().sum() # 查看合并后是否存在空值


# In[160]:


df4.info()# 数据概览


# # Step 5: 特征衍生

# In[244]:


df5=df4.copy()
df5.info()


# In[81]:


# 绘制连续变量直方图/QQ图
con_col5= list(df5.select_dtypes(exclude=object))
cols = 6
rows = len(con_col5)-1
plt.figure(figsize=(4*cols,4*rows))

from scipy import stats
i=5
for c in con_col5:
    i+=1
    ax=plt.subplot(rows,cols,i)
    sns.distplot(df5[c],fit=stats.norm,color='navy')
    i+=1
    ax=plt.subplot(rows,cols,i)
    res = stats.probplot(df5[c], plot=plt)
plt.show()


# In[163]:


# Kmeans 聚类
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')

# 利用手肘法选取最优聚类数目
wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df5[con_col5].values)
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 10), wcss,color='crimson')
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[246]:


# 根据上图确定n_clusters=4
model = KMeans(n_clusters=4, init='k-means++', n_init=100, max_iter=1000)
km_clusters = model.fit_predict(df5[con_col5].values)
km_clusters


# In[247]:


df5['km_clusters']=km_clusters # 聚类结果加入数据集


# In[250]:


# 将相关系数较高的连续性变量与文本型变量进行交叉衍生
X_train1=df5[df5['train_or_test']==0]

con_col1= list(df5.select_dtypes(exclude=object))
obj_col1= list(df5.select_dtypes(include=object))
high_con_col=[x for x in con_col1 if abs(X_train1[x].corr(Y_train['jieju_dubil_status_desc']))>0.1]

# 连续变量按不同类别计数
df6=df5.copy()
for i in obj_col1:
    for j in high_con_col:
        cross=pd.crosstab(df6[i],df6[j])
        df6[i+'with'+j]=df6[i]
        df6[i+'with'+j].replace(sorted(df6[i+'with'+j].unique()),cross.apply(lambda x: x.sum(), axis=1),inplace=True)

# 类别与类别间计数
df7=df6.copy()
for i in obj_col1:
    for j in obj_col1:
        if i!=j:
            cross=pd.crosstab(df7[i],df7[j])
            df7[i+'with'+j]=df7[i]
            df7[i+'with'+j].replace(sorted(df7[i+'with'+j].unique()),cross.apply(lambda x: x.sum(), axis=1),inplace=True)
df7.head()


# In[171]:


df8=df7.drop(columns=[x for x in df7.select_dtypes(include=object).columns])
X_train8=df8[df8['train_or_test']==0]

high_corr8 = [i for i in df8.columns if abs(X_train8[i].corr(Y_train['jieju_dubil_status_desc']))>0.1]

# 挑选相关性较高的特征做聚类
wcss = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df8[high_corr8].values)
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 15), wcss,color='crimson')
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[186]:


# 根据上图确定n_clusters=10
model = KMeans(n_clusters=10, init='k-means++', n_init=100, max_iter=1000)
km_clusters = model.fit_predict(df8[high_corr8].values)
km_clusters


# In[174]:


# 聚类结果可视化
def plot_clusters(samples, clusters):
    for sample in range(len(clusters)):
        x=samples[sample][0]
        y=samples[sample][1]
        plt.scatter(x,y)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()
plot_clusters(df8[high_corr8].values, km_clusters)


# In[187]:


df8['km_clusters']=km_clusters # 聚类结果加入数据集


# In[188]:


for i in high_corr8:
    df8[i+'_std']=np.sqrt(df8[i])
    df8[i+'_square']=df8[i]*df8[i]
    df8[i+'_exp']=np.exp(df8[i])
    df8[i+'_log'] = np.log(df8[i])
    try:
        df8['1/'+i]=1/df8[i]
    except:
        pass
    for j in high_corr8:
        if i!=j:
            df8[i+j]=df8[i]+df8[j]
            df8[i+'*'+j]=df8[i]*df8[j]
            try:
                df8[i/j]=df8[i]/df8[j]
                df8[j/i]=df8[j]/df8[i]
            except:
                pass
df8.head()


# In[189]:


print(df8.isnull().sum().sum())
np.isinf(df8).sum().sum()


# In[190]:


#缺失值处理
for i in list(df8.select_dtypes(exclude = object)):
    df8[i].fillna(df8[i].median(), inplace=True)  # 中位数填充

# 无穷值处理
df8 = df8.replace(-np.inf,10e-6).replace(np.inf,10e+6)

# 检查填充情况
print(df8.isnull().sum().sum())
np.isinf(df8).sum().sum()


# In[191]:


df8.info()


# In[193]:


# 缺失值、无穷值处理同上
for i in list(df8.select_dtypes(exclude = object)):
    df8[i].fillna(df8[i].median(), inplace=True)   
    
df8 = df8.replace(-np.inf,10e-6).replace(np.inf,10e+6)

print(df8.isnull().sum().sum())
np.isinf(df8).sum().sum()


# In[197]:


#数据标准化处理： 区间缩放法
from sklearn.preprocessing import MinMaxScaler
 
#区间缩放，返回值为缩放到[0, 1]区间的数据
df9= MinMaxScaler().fit_transform(df8)
df10=pd.DataFrame(df9,columns=list(df8.columns))
df10.info()


# In[ ]:


# 取相关系数较高的两列的最大值/均值
X_train10 = df10[df10['train_or_test']==0]
high_corr10 = [i for i in df10.columns if abs(X_train10[i].corr(Y_train['jieju_dubil_status_desc']))>0.2]
for i in high_corr10:
    high_corr10.remove(i)
    for j in high_corr10:
        df10['mean_'+i+'_'+j] = df10[[i,j]].mean(axis=1)
        df10['max_'+i+'_'+j] = df10[[i,j]].max(axis=1)
df10.shape


# In[184]:


# 数理聚合
df10['sum']=df10.apply(lambda x: x.sum(), axis=1) # 和
df10['mean']=df10.apply(lambda x: x.mean(), axis=1) # 均值
df10['std']=df10.apply(lambda x: x.std(), axis=1) # 标准差
df10['mean/std']=df10.apply(lambda x: x.mean()/x.std(), axis=1) # 变异系数
df10['max']=df10.apply(lambda x: x.max(), axis=1) # 最大值
df10['min']=df10.apply(lambda x: x.min(), axis=1) # 最小值
df10['max-min']=df10.apply(lambda x: x.max()-x.min(), axis=1) # 极差


# In[199]:


df10.head()


# In[200]:


# 检查缺失值和无穷值
print(df10.isnull().sum().sum())
np.isinf(df10).sum().sum()


# In[213]:


#数据标准化处理： 区间缩放法
from sklearn.preprocessing import MinMaxScaler
 
#区间缩放，返回值为缩放到[0, 1]区间的数据
df11= MinMaxScaler().fit_transform(df10)
df12=pd.DataFrame(df11,columns=list(df10.columns))
df12.head()


# In[214]:


# 检查缺失值和无穷值
print(df12.isnull().sum().sum())
np.isinf(df12).sum().sum()


# # Step 6: 建立模型

# In[215]:


df=df12.copy()


# In[216]:


# 压缩数据大小,以使部分复杂的算法能顺利进行
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum()/1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()/1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100*(start_mem - end_mem) / start_mem))
    return df
reduce_mem_usage(df)


# In[217]:


df.shape


# In[218]:


df['train_or_test'].value_counts()


# In[219]:


# 拆分训练集和预测集
train = df[df['train_or_test']<1].drop('train_or_test',axis=1,inplace=False)
test = df[df['train_or_test']>0].drop('train_or_test',axis=1,inplace=False)
y=Y_train
# 按7:3的比例将划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2,random_state=42) 

print(train.shape)
print(test.shape)
print(y.shape)


# LightGBM

# In[134]:


# 调整参数

# step1:确定最优n_estimators
from sklearn.datasets import load_breast_cancer
params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1,
          'num_leaves':30, 
          'max_depth': 5,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
    }
data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())


# In[141]:


# step2:确定最优max_depth和num_leaves
params_test1={'max_depth': range(3,8,1), 'num_leaves':range(5, 100, 5)}
estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                               objective='binary',
                               metrics='auc',
                               learning_rate=0.1,
                               n_estimators=5000)
gsearch1 = GridSearchCV(estimator, param_grid = params_test1, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch1.fit(X_train,y_train)
gsearch1.cv_results_,gsearch1.best_params_,gsearch1.best_score_


# In[ ]:


# step3:确定min_data_in_leaf和max_bin
params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                               objective='binary',
                               metrics='auc',
                               learning_rate=0.1, 
                               n_estimators=5000, 
                               max_depth=40, 
                               num_leaves=50)

gsearch2 = GridSearchCV(estimator,param_grid = params_test2, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch2.fit(X_train,y_train)
print(gsearch2.cv_results_)
print(gsearch2.best_params_)
print(gsearch2.best_score_)


# In[ ]:


# step4:确定feature_fraction、bagging_fraction、bagging_freq
params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': range(0,81,10)
}
estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                               objective='binary',
                               metrics='auc',
                               learning_rate=0.1, 
                               n_estimators=5000, 
                               max_depth=40, 
                               num_leaves=50,)
              
gsearch3 = GridSearchCV(estimator, param_grid = params_test3, scoring='roc_auc',cv=5，n_jobs=-1)
gsearch3.fit(X_train,y_train)
print(gsearch3.cv_results_)
print(gsearch3.best_params_)
print(gsearch3.best_score_)


# In[ ]:


# step5:确定 min_split_gain 
params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
              
gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                                       objective='binary',
                                                       metrics='auc',
                                                       learning_rate=0.1, 
                                                       n_estimators=5000, 
                                                       max_depth=40, 
                                                       num_leaves=50,
                                                       max_bin=15,
                                                       min_data_in_leaf=51,
                                                       bagging_fraction=0.6,
                                                       bagging_freq= 0.8, 
                                                       feature_fraction= 0.8,
                                                       lambda_l1=1e-05,
                                                       lambda_l2=1e-05), 
                       param_grid = params_test5, scoring='roc_auc',cv=5,n_jobs=-1)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[223]:


# step6:降低学习率，增加迭代次数，验证模型
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         metrics='auc',
                         learning_rate=0.01, 
                         n_estimators=5000, 
                         max_depth=40, 
                         num_leaves=50,
                         max_bin=15,
                         min_data_in_leaf=51)
lgb_model.fit(X_train,y_train)
prediction_lgb = lgb_model.predict_proba(X_test)[:,1]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test,prediction_lgb)
roc_auc = metrics.auc(fpr_lgb, tpr_lgb)
print('model score:',lgb_model.score(X_test,y_test)) 
print('AUC:',metrics.roc_auc_score(y_test,prediction_lgb))
print('ave:',(lgb_model.score(X_test,y_test)+metrics.roc_auc_score(y_test,prediction_lgb))/2)


# In[251]:


# step7:继续手动调整参数
# 注：此过程经历数次尝试，代码同step6，只改变具体参数值，故已省略中间过程
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         metrics='auc',
                         learning_rate=0.01, 
                         n_estimators=10000, 
                         max_depth=40, 
                         num_leaves=50,
                         max_bin=15,
                         min_data_in_leaf=20)
lgb_model.fit(X_train,y_train)
prediction_lgb = lgb_model.predict_proba(X_test)[:,1]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test,prediction_lgb)
roc_auc = metrics.auc(fpr_lgb, tpr_lgb)
print('model score:',lgb_model.score(X_test,y_test))
print('AUC:',metrics.roc_auc_score(y_test,prediction_lgb))
print('ave:',(lgb_model.score(X_test,y_test)+metrics.roc_auc_score(y_test,prediction_lgb))/2)


# In[225]:


# 绘制ROC、AUC曲线
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_lgb, tpr_lgb, 'b', label = 'AUC = %0.2f' % roc_auc,color='purple')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# GBDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
param_distributions = {'n_estimators': [200,400],
                       'max_depth': [1,3,5],
                       'subsample': [0.2,0.5,0.8], 
                       'learning_rate': [0.02,0.04,0.06]
                       }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
gbc = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=1),
              param_distributions=param_distributions, n_iter=10, cv=kfold, random_state=1)
gbc.fit(X_train, y_train)
gbc.best_params_

## 代入最优参数
gbc = gbc.best_estimator_
gbc.fit(X_train,y_train)
prediction_gbc= gbc.predict_proba(X_test)
roc_auc_score(y_test,prediction_gbc[:,1])
fpr_gbc, tpr_gbc, _ = roc_curve(y_test,prediction_gbc[:,1])


# In[ ]:


# XGBoost
xgb_model = xgb.XGBClassifier(random_state=1)
param_dict={'max_depth':[1,2,3],
            'n_estimators':[100,200,500,800],
            'learning_rate':[0.01,0.03,0.05]
            }
clf=GridSearchCV(xgb_model,param_dict,scoring='roc_auc',verbose=1)
clf.fit(X_train,y_train)

## 代入最优参数
xgbc= XGBClassifier(learning_rate=0.03,max_depth=2,n_estimators=500,random_state=1)
xgbc.fit(X_train, y_train)
prediction_xgbc= xgbc.predict_proba(X_test)
roc_auc_score(y_test,prediction_xgbc[:,1])
fpr_xgbc, tpr_xgbc, _ = roc_curve(y_test,prediction_xgbc[:,1])


# In[ ]:


# 统一绘制不同模型ROC曲线
plt.plot(fpr_lgb, tpr_lgb, label='LightGBM')
plt.plot(fpr_gbc, tpr_gbc, label='GBDT')
plt.plot(fpr_xgbc, tpr_xgbc, label='XGBoost')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[227]:


# 选择相对最优的lgb模型，统计特征重要性
iy = lgb_model.feature_importances_
feature_importance = pd.DataFrame()
feature_importance['fea_name'] = X_train.columns
feature_importance['fea_imp'] = lgb_model.feature_importances_
f=feature_importance.sort_values(by='fea_imp', ascending=False)[:50]
f.sort_values(by='fea_imp', ascending=True,inplace=True)
f.reset_index(drop=True,inplace=True)
f


# In[228]:


# 特征重要性可视化
plt.figure(figsize=(10,10))
plt.barh(f.fea_name, f.fea_imp,color='purple')
plt.xlabel('Feature Importance',fontsize=15)
plt.ylabel('Feature')
plt.title('Feature Importance of LightGBM',fontsize=20)
plt.tight_layout()


# In[229]:


# 选取特征重要性高于500的特征
high_f1=f[f.fea_imp>1000].fea_name
high_f1


# In[230]:


df13=df12.copy()
df13.info()


# In[231]:


df13[high_f1]


# In[234]:


# 绘制连续变量直方图/QQ图
from scipy import stats
con_col13= list(df13[high_f1].select_dtypes(exclude=object))
cols = 6
rows = len(con_col13)-1
plt.figure(figsize=(4*cols,4*rows))

i=0
for c in con_col13:
    i+=1
    ax=plt.subplot(rows,cols,i)
    sns.distplot(df13[c],fit=stats.norm,color='navy')
    i+=1
    ax=plt.subplot(rows,cols,i)
    res = stats.probplot(df13[c], plot=plt)
plt.show()


# In[237]:


# 计数编码
df14=df13.copy()
for i in high_f1:
    df14[i+'_C'] = LabelEncoder().fit_transform(df14[i]) 
df14.head()


# In[238]:


#数据标准化处理： 区间缩放法
from sklearn.preprocessing import MinMaxScaler
 
#区间缩放，返回值为缩放到[0, 1]区间的数据
df15= MinMaxScaler().fit_transform(df14)
df16=pd.DataFrame(df15,columns=list(df14.columns))
df16.head()


# In[209]:


df=df16.copy()


# In[239]:


# 拆分训练集和预测集
train = df[df['train_or_test']<1].drop('train_or_test',axis=1,inplace=False)
test = df[df['train_or_test']>0].drop('train_or_test',axis=1,inplace=False)
y=Y_train
# 按7:3的比例将划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2,random_state=42) 

print(train.shape)
print(test.shape)
print(y.shape)


# In[240]:


# LightGBM
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         metrics='auc',
                         learning_rate=0.01, 
                         n_estimators=5000, 
                         max_depth=15, 
                         num_leaves=50,
                         max_bin=15,
                         min_data_in_leaf=20)
lgb_model.fit(X_train,y_train)
prediction_lgb = lgb_model.predict_proba(X_test)[:,1]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test,prediction_lgb)
roc_auc = metrics.auc(fpr_lgb, tpr_lgb)
print('model score:',lgb_model.score(X_test,y_test)) #在测试集上看性能
print('AUC:',metrics.roc_auc_score(y_test,prediction_lgb))
print('ave:',(lgb_model.score(X_test,y_test)+metrics.roc_auc_score(y_test,prediction_lgb))/2)


# In[224]:


# 综合考虑model score,auc和average 三个指标选择最优结果
# 运用模型在训练集上预测
prediction_lgb= lgb_model.predict_proba(test)[:,1] 

#将结果写入output
output['LABEL']=prediction_lgb

#将结果写入文件
with pd.ExcelWriter("结果提交.xlsx") as writer:
    output.to_excel(writer,sheet_name="sheet1",index=False)  

