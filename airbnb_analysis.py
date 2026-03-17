import numpy as np;
import pandas as pd;
import seaborn as sns;
from google.colab import drive;
drive.mount('/content/drive');
df = pd.read_csv('/content/drive/MyDrive/Airbnb.csv')
df.head()
df['host Certification'].dtypes
df.isnull().sum()
df.drop('host Certification',axis=1,inplace=True)
cols = ['price','consumer']
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['consumer'] = pd.to_numeric(df['consumer'], errors='coerce')
for col in cols:
    df[col] = df[col].fillna(df[col].mean()) # Fixed FutureWarning
df.drop('reply time',axis=1,inplace=True)
df.isnull().sum()
df.duplicated().sum()
df.describe()
Q1=df['price'].quantile(0.25)
Q3=df['price'].quantile(0.75)
IQR=Q3-Q1
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
df['price']=df['price'].clip(lower=lower_limit,upper=upper_limit)
sns.boxplot(df['price'])
df.drop(['id','name','host_id','host_name','host since'],axis=1,inplace=True)
df['host response rate'] = pd.to_numeric(df['host response rate'].str.replace(',','.'),errors='coerce')
df['host acceptance rate'] = pd.to_numeric(df['host acceptance rate'].str.replace(',','.'),errors='coerce')
df['bathrooms']=df['bathrooms'].str.replace(',','.')
df['bathrooms'] = df['bathrooms'].str.extract(r'(\d+\.?\d*)')[0].astype(float) # Fixed SyntaxWarning
df=pd.get_dummies(df,columns=['city'],drop_first=True)
df=pd.get_dummies(df,columns=['area'],drop_first=True)
df.select_dtypes(include=['object']).columns

# Define features (x) and target (y) before splitting
y = df['price']
x = df.drop('price', axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler # Corrected typo: standaradscalar to StandardScaler
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
from sklearn.linear_model import LinearRegression # Corrected typo: linearregression to LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

sns.histplot(df['price'],kde=True)
plt.title("price distribution")
plt.show()

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("correlation matrix")
plt.show()

sns.scatterplot(x='consumer', y='price', data=df)
plt.title("Consumer vs Price")
plt.show()

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


