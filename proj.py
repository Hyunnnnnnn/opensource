from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib import font_manager, rc
font_path = "./malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

url = 'https://www.index.go.kr/unify/idx-info.do?idxCd=4259'
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'html.parser')

col = soup.select('div > table > thead > tr > th')
names = soup.select('div > table > tbody > tr > th')
figs = soup.select('div > table > tbody > tr > td')

df_columns = []
for i in col:
    df_columns.append(i.get_text(strip=True))
df_columns = list(filter(None, df_columns))

df_idx = []
for i in names:
    df_idx.append(i.get_text())

fig = []
df_value = []
count = 0
for i in figs:
    fig.append(i.get_text())
    count += 1
    if count == 13:
        df_value.append(fig)
        count = 0
        fig = []

df = pd.DataFrame(df_value, index=df_idx, columns=df_columns)

df['2008'] = df['2008'].astype(float)
df['2009'] = df['2009'].astype(float)
df['2010'] = df['2010'].astype(float)
df['2011'] = df['2011'].astype(float)
df['2012'] = df['2012'].astype(float)
df['2013'] = df['2013'].astype(float)
df['2014'] = df['2014'].astype(float)
df['2015'] = df['2015'].astype(float)
df['2016'] = df['2016'].astype(float)
df['2017'] = df['2017'].astype(float)
df['2018'] = df['2018'].astype(float)
df['2019'] = df['2019'].astype(float)
df['2020'] = df['2020'].astype(float)
df.to_excel('./public_transport.xlsx')

ndf = df.drop(['대중교통'])
ndf['대중교통'] = True, True, False, False
print(df)


grouped = ndf.groupby(['대중교통'])
for key, group in grouped:
    print('대중교통:', key)
    print('* number: ', len(group))
    print(group)
    print('\n')

average = grouped.mean()
print(average)


def min_max(x):
    return x.max() - x.min()


agg_sep = grouped.agg({'2008':['min','max','sum'], '2019':min_max})
print(agg_sep)

plt.style.use('ggplot')

tdf = df.T
tdf.plot(kind='bar', width=0.5)
plt.title('교통수단별 수송분담률')
plt.xlabel('연도')
plt.ylabel('수송분담률')
plt.show()

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

ax1.plot(df_columns, df.loc['철도',:], color='green', label='철도')
ax2.plot(df_columns, df.loc['버스',:], color='skyblue', label='버스')
ax3.plot(df_columns, df.loc['택시',:], color='yellow', label='택시')
ax4.plot(df_columns, df.loc['승용차',:], color='olive', label='승용차')

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')

ax1.set_title('철도 수송분담률', size=15)
ax2.set_title('버스 수송분담률', size=15)
ax3.set_title('택시 수송분담률', size=15)
ax4.set_title('승용차 수송분담률', size=15)
plt.show()

x=tdf[['버스']]
y=tdf[['승용차']]


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=10)

from  sklearn.linear_model import LinearRegression


lr = LinearRegression()

lr.fit(x_train, y_train)

print('기울기 a : ', lr.coef_)
print('y절편 b : ', lr.intercept_)

y_hat = lr.predict(x)

plt.figure(figsize=(10,5))
ax1 = sns.histplot(y,kde=True,label='y',color='red')
ax2 = sns.histplot(y_hat,kde=True, label="y_hat",color='blue' ,ax=ax1)
plt.legend()



plt.show()
