import numpy as np
import pandas as pd
import math
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


df = pd.read_csv('census (3).csv')
#### first change dates year to 1994
print(df.columns)
###############
# string length for normal format MM/DD/YYYY
##length_str = len(df.iloc[1,0])
### problem 1
##print(df.iloc[:,0])

df.iloc[2,0] = "1/21/1994"
##print(df.iloc[:,0])
###5001 to 8000
df.iloc[(5001-2):(8001-1),0] = "1/30/1994"
df.iloc[(32529-2):(32549-1),0] = "5/7/1994"
df.iloc[32557,0] = "5/1/1994"
df.iloc[32558,0] = "2/14/1994"
df.iloc[32559,0] = "2/14/1994"
df.iloc[32560,0] =  "2/14/1994"
df.iloc[0,0] = "5/1/1994"

### problem 2
##print(df.iloc[:,1])
## 10 being lower age < 20
## and 100 being the highest age group
def Set_age(upper,lower):

    df.loc[(df.age<upper) & ((lower<df.age) | (df.age == lower)),"age"] = lower





df.loc[df.age<20,"age"] = 10 ### 10 is equvalent to 10's age interval
Set_age(30,20)
Set_age(40,30)
Set_age(50,40)
Set_age(60,50)
Set_age(70,60)
Set_age(80,70)
Set_age(90,80)
Set_age(100,90)
Set_age(110,100)




### problem 3

##df['workclass'] = df['workclass'].replace(['?'],'other')
##df['occupation'] = df['occupation'].replace(['?'],'other')

##print("?" == df.loc[:,['workclass','occupation']])
df['workclass'] = df['workclass'].str.strip()      ### Hey Look here
df['occupation'] = df['occupation'].str.strip()    ### This strips columns with strings with white space so that you can replace them
df['workclass'] = df['workclass'].replace(['?'],'Other')
df['occupation'] = df['occupation'].replace(['?'],'Other')



### problem 4
colum_population_wgt= "population-wgt"
max_value = df[colum_population_wgt].max()
min_value = df[colum_population_wgt].min()
df[colum_population_wgt] = (df[colum_population_wgt]-min_value)/(max_value-min_value)


#### problem 5
df['sex'] = df['sex'].str.strip()
print(df.loc[(df.sex != 'Male') & (df.sex != 'Female'),'sex']) ### used to check all possable to be removed
df['sex'] = df['sex'].replace(['m'],'Male')
df['sex'] = df['sex'].replace(['M'],'Male')
df['sex'] = df['sex'].replace(['F'],'Female')
df['sex'] = df['sex'].replace(['male'],'Male')
df['sex'] = df['sex'].replace(['female'],'Female')
df['sex'] = df['sex'].replace(['fem'],'Female')
df['sex'] = df['sex'].replace(['f'],'Female')

### problem 6
print("max","min")
print(df["hours-per-week"].max(),df["hours-per-week"].min()) ### determine max and men

df["hours-per-week"] = np.ceil((df["hours-per-week"])/20).astype('int')

print(df['hours-per-week'])



### problem 7

df['native-country'] = df['native-country'].str.strip()

df['native-country'] = df['native-country'].replace(['?'],'Unspecified')



### problem 9 chi^2 test
def Chi_square_test(data_frame):
  ##dof = (row-1)*(cols-1)
  sig = 0.05
  chi, pval, dof, exp = chi2_contingency(data_frame)
  p = 1 - sig
  critical_value = chi2.ppf(p,dof)


  if(chi > critical_value):

      return "N"

  else:

      return "I"

atribut_list = ['age','workclass','education','marital-status','occupation'\
,'relationship','race','sex','hours-per-week','native-country']
Chi_fram = pd.DataFrame( columns = atribut_list, index=atribut_list)

X = df.copy()
###X1 = X.groupby(['age','workclass'])[['workclass']].count().unstack()

V = []
for i in range(0,len(atribut_list)):

   for j in range(i+1,len(atribut_list)):

        Add_list = X.groupby([atribut_list[i],atribut_list[j]])[[atribut_list[j]]].count().unstack().fillna(0)
        V.append(Add_list)

count = 0;
for i in range(0,len(atribut_list)):

   for j in range(i+1,len(atribut_list)):



        Chi_fram.iloc[i,j] = Chi_square_test(V[count])
        count = count + 1


Chi_fram.to_csv('I_test.csv')

#### problem 10

def spear_val(row_index, col_index,DataFrame_1,list_value):

    X1 =  DataFrame_1.loc[:,list_value[row_index]].values.reshape(-1,1)
    Y1 =  DataFrame_1.loc[:,list_value[col_index]].values.reshape(-1,1)
    corr, p_value = spearmanr(X1,Y1)





    if( (np.abs(corr) >= 0.8)):


        return "N"


    else:


        return "I"





    ##return corr

atribut_list2 = ['date','population-wgt','education-num','capital-gain','capital-loss']
spear_fram = pd.DataFrame( columns = atribut_list2, index=atribut_list2)

for i in range(0,len(atribut_list2)):

    for j in range(i+1,len(atribut_list2)):

        spear_fram.iloc[i,j] = spear_val(i,j,X,atribut_list2)



print(spear_fram)
spear_fram.to_csv("spea_fram.csv")


######## problem 10  ####
Y = df.copy()
Y.to_csv('census2.csv')


n_componets = 6
pca = PCA(n_components=n_componets)
reduced = pca.fit_transform(df[["age","population-wgt",\
"education-num",\
"capital-gain",'capital-loss','hours-per-week']])

for i in range(0,n_componets):
    df['PC' + str(i+1)] = reduced[:,i]

PC = []
for i in range(0,len(pca.components_)):
    x = pca.components_[0,i]
    y = pca.components_[1,i]
    if x > 0 and y > 0:
        qud = 1
    elif x < 0 and y > 0:
        qud = 3
    elif x< 0 and y < 0:
        qud = 2
    else:
        qud = 1

    PC.append((qud,x,y,df.columns.values[i]))

sorted = sorted(PC, key= lambda tup: (tup[0],tup[1]),reverse=True)
for i in sorted:
    print(i)
