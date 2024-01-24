import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=Warning)
pd.set_option("display.width",500)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import matplotlib
matplotlib.use("Qt5Agg")

cars = pd.read_csv("free_work/car_price_prediction/car_price_prediction.csv")
df = cars.copy()

#--------------------------------- Data Preprocessing ----------------------------------------------

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

df.columns = [col.replace(" ", "_").upper() for col in df.columns]
del df["ID"]


#--------------------------------- Feature Engineering ----------------------------------------------

# Transformation Variable Types

# LEVY
df["LEVY"] = [val.replace("-", "0") for val in df["LEVY"].values]
df["LEVY"] = df["LEVY"].astype(int)


# ENGINE_VOLUME
# motor hacmi değişkeninde turbo eklentileri oldugu icin onları extra bir turbo var-yok değişkenine atıcam.
turbo = []
for i in range(len(df)):
    if "Turbo" in df["ENGINE_VOLUME"].values[i]:
        turbo.append(1)
    else:
        turbo.append(0)

# içindeki turboları silme işlemi
df["ENGINE_VOLUME"] = [col.replace("Turbo", "").upper() for col in df["ENGINE_VOLUME"].values]
df["ENGINE_VOLUME"] = df["ENGINE_VOLUME"].astype(float)

# turbo var-yok ekleme
df.insert(10,column="TURBO",value=turbo)
df.iloc[20:25,:]["TURBO"]

# MILEAGE
df["MILEAGE"] = [val.replace("km", "").upper() for val in df["MILEAGE"].values]
df["MILEAGE"] = df["MILEAGE"].astype(int)


# ------------------------ Handling Outlier Values ------------------------------------------


# Deleting 0 km vehicles
# 0 km araçlar ya yeni araçlardır ki o zaman bu data setinde bulunması anlamsızdır çünkü yeni araçların fiyatları sabittir tahmin edilemez eğer yanlış girilmiş ise fiyatı doğrudan etkileyen bir değişken olduğundan aykırı değer olarak kabul edilebilir
df.drop(index=df.loc[df["MILEAGE"]==0].index,axis=0,inplace=True)
df.loc[df["MILEAGE"]==0]

# km si 1 milyonun üstünde olanlar
df.sort_values(by="MILEAGE",ascending=False).head(40)
df.drop(index=df.loc[df["MILEAGE"]>1000000].index,axis=0,inplace=True)

# km si 1000 in altında olanlar
df.drop(index=df.loc[df["MILEAGE"]<1000].index,axis=0,inplace=True)

# data setindeki araç modeli sayısının 10 dan küçük olanların silinmesi
df["MANUFACTURER"].value_counts()

# Yüzdesel gösterim
(100*(df["MANUFACTURER"].value_counts())/18507)

# 0.05 inden küçük olanlar
((100*(df["MANUFACTURER"].value_counts())/18507)<0.05).sum() # 24 tane

# 0.05 inden küçük olanların silinmesi
for i in df["MANUFACTURER"].unique():
    if (100*(df.loc[df["MANUFACTURER"]==i,"MANUFACTURER"].count())/18507)< 0.05:
        df.drop(df.loc[df["MANUFACTURER"] == i, "MANUFACTURER"].index,axis=0,inplace=True)

df["MANUFACTURER"].value_counts()



# fiyatı aşırı yüksek olanlar
df.sort_values(by="PRICE",ascending=False).head()
# markası OPEL Combo 1999 model olan aracın fiyatı 26307500 birim olması imkansız
df.drop([16983],axis=0,inplace=True)

# fiyatı aşırı düşük olanlar
df.sort_values(by="PRICE",ascending=True).head(150)

# Fiyatı 1000 birimden kucuk olan araçların silinmesi
df.drop(df.loc[df["PRICE"]<1000].index,axis=0,inplace=True)



# data setindeki araç model yılları sayıları 10 dan küçük olanlar
df["PROD._YEAR"].value_counts()

# data setindeki araç model yılları sayılarının 10 dan küçük olanlarının silinmesi
for i in df["PROD._YEAR"].unique():
    if (df.loc[df["PROD._YEAR"]==i,"PROD._YEAR"].count())< 10:
        df.drop(df.loc[df["PROD._YEAR"] == i, "PROD._YEAR"].index,axis=0,inplace=True)

# model yılları artık 1988-2020 arasında
df["PROD._YEAR"].value_counts()

# transforming PROD._YEAR to cars` age
df["CAR_AGE"] = 2023 - df["PROD._YEAR"]
del df["PROD._YEAR"]
df.head()


# ---------------------------------------------- EDA -------------------------------------------
# ---------------------------------- Data Analysis & Visualization -------------------------------

# vites türlerine göre ort araç fiyatları
sns.barplot(x="GEAR_BOX_TYPE",y="PRICE",data=df,errwidth=0,palette="magma").set_title('MEAN OF GEAR_BOX_TYPE BY PRICE')
# tiptronik daha pahalı oldugu görülüyor

# araç sınıflarına göre ort araç fiyatları
plt.figure(figsize=(10,8))
sns.barplot(x="CATEGORY",y="PRICE",data=df,errwidth=0,palette="magma").set_title('MEAN OF PRICE BY CAR CLASS')
plt.tight_layout()

# araç renklerine göre ort araç fiyatları
sns.barplot(x="PRICE",y="COLOR",data=colors,errwidth=0,palette="magma").set_title('MEAN OF PRICE BY CAR COLOR')
plt.tight_layout()
# çok dikkat çekici renklerin daha düşük fiyatlı olduğu görülüyor. like pink, orange

# If the vehicles have a turbo, the prices increase by an average of 10,000.
sns.barplot(x="TURBO",y="PRICE",data=df,errwidth=0,palette="magma").set_title('MEAN OF PRICE BY TURBO')


# levy ve price ın drive_wheels e göre
fig, ax = plt.subplots(1,2,figsize=(12,6))
fig.suptitle('MEAN OF PRICE & LEVY BY DRIVE_WHEELS')
sns.barplot(x="DRIVE_WHEELS",y="LEVY",data=df,errwidth=0,palette="magma",ax=ax[0])
ax[0].set_title('MEAN OF LEVY BY DRIVE_WHEELS')
sns.barplot(x="DRIVE_WHEELS",y="PRICE",data=df,errwidth=0,palette="magma",ax=ax[1])
ax[1].set_title('MEAN OF PRICE BY DRIVE_WHEELS')
plt.tight_layout()

# araçlar daha çok ortalama 4 silindirli diyebiliriz
sns.countplot(x="CYLINDERS",data=df,palette="magma").set_title('AVERAGE NUMBER OF CYLINDERS BY CARS')


# --------------------------- Correlation Analysis & Pairplot & Kdeplot ----------------------------
# Kde Plot
sns.pairplot(df, diag_kind="kde",height=1.5)

# Heatmap
plt.figure(figsize= (12,8))
df.corr(numeric_only=True).sort_values(by="PRICE",ascending=False)
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.tight_layout()
plt.show()


# I want to look at the mileage and price change after handling outliers. Before looking at kde and bloxplot we need to convert the previous mileage variable
df_before = cars.copy()
df_before.head()

df_before["Mileage"] = [val.replace("km", "").upper() for val in df_before["Mileage"].values]
df_before["Mileage"] = df_before["Mileage"].astype(int)


# Boxplot
fig, axes = plt.subplots(2,2, figsize=(10, 8))

axes[0,0].boxplot(df_before["Mileage"])
axes[0,1].boxplot(df["MILEAGE"])
axes[1,0].boxplot(df_before["Price"])
axes[1,1].boxplot(df["PRICE"])

axes[0, 0].set_title('Before Mileage')
axes[0, 1].set_title('After Mileage')
axes[1, 0].set_title('Before Price')
axes[1, 1].set_title('After Price')

plt.suptitle('BOXPLOT of MILEAGE & PRICE', fontsize=16)
plt.tight_layout()
plt.show()


# kdeplot
fig, axes = plt.subplots(2,2, figsize=(10, 8))

sns.kdeplot(df_before["Mileage"], fill=True, ax=axes[0,0])
sns.kdeplot(df["MILEAGE"], fill=True, ax=axes[0,1])
sns.kdeplot(df_before["Price"], fill=True, ax=axes[1,0])
sns.kdeplot(df["PRICE"], fill=True, ax=axes[1,1])

axes[0, 0].set_title('Before Mileage')
axes[0, 1].set_title('After Mileage')
axes[1, 0].set_title('Before Price')
axes[1, 1].set_title('After Price')

fig.suptitle('DISTRIBUTION of MILEAGE & PRICE', fontsize=16)
plt.tight_layout()
plt.show()

# When we look at the graphs, they actually look good, they can be done better, but I don't want to interfere with the data anymore


# ----------------------------------- Encoding  --------------------------------------------
# One Hot Encoding
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_cols[2:]
df2 = df.copy()
df2 = pd.get_dummies(df2,columns=cat_cols[2:],drop_first=True,dtype=int)


# --------------------------------------------- MODELLEME -----------------------------------------------

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor



x = df2.drop(["PRICE","MANUFACTURER","MODEL"], axis=1)
y = df2[['PRICE']]


models = [('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("LightGBM", LGBMRegressor(verbose=-1))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 8889.2095 (RF)
# RMSE: 10982.2137 (GBM)
# RMSE: 9315.6175 (LightGBM)

# -------------------------------------- effect of transform LOG ----------------------------------------
# I want to look PRICE variable by log transform
from scipy import stats

stats.skew(df2["PRICE"])

# for PRICE transformation log
trans_price = np.log1p(df2["PRICE"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(ax=ax1, data=df["PRICE"])
ax1.set_title("Raw Data")

sns.histplot(ax=ax2, data=trans_price)
ax2.set_title("Log Transformation")

plt.show()

before = stats.skew(df["PRICE"])
after = stats.skew(trans_price)
print(f"before log transform: {before}\nafter log transform :{after}")

# Let's look at the difference between before and after with probplot
ax3 = plt.subplot(121)
stats.probplot(df["PRICE"], plot=plt)

ax4 = plt.subplot(122)
stats.probplot(trans_price, plot=plt)

plt.suptitle('PROBPLOT of PRICE', fontsize=16)
plt.tight_layout()
plt.show()

# We can say that the TARGET variable is approaching normal

# ----------------------------------------- applicating LOG to model ------------------------------
# i choose random forest regressor according to rmse results

x = df2.drop(["PRICE","MANUFACTURER","MODEL"], axis=1)
y = np.log1p(df2['PRICE'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)

rf = RandomForestRegressor().fit(x_train, y_train)
y_pred = rf.predict(x_test)

inverse_ypred = np.expm1(y_pred)
inverse_y_test = np.expm1(y_test)

print(np.sqrt(mean_squared_error(inverse_y_test, inverse_ypred)))
# 11609.971
rf.score(x_test,y_test) # r2_score(y_test,y_pred) aynısı.
# 0.8066118731178298

# ------------------------------------------ MODEL OPTIMIZATION ----------------------------------

x = df2.drop(["PRICE","MANUFACTURER","MODEL"], axis=1)
y = df2[['PRICE']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
rf = RandomForestRegressor(random_state=46)


parameters = {
    'n_estimators': range(100,500),
    'max_features':['sqrt', 'log2', None],
    'bootstrap':[True, False],
    'min_samples_split':range(2,20)}

rf_params = {"min_samples_split": [2,3,4,5,6, 8],
             "n_estimators": [105,110, 115, 120]}




rf_best = RandomizedSearchCV(estimator=rf,
                        param_distributions=parameters,
                        n_iter = 20,
                        cv=3,
                        n_jobs=-1,
                        verbose=2).fit(x, y)

"""
{'n_estimators': 114,
 'min_samples_split': 5,
 'max_features': None,
 'bootstrap': True}
"""
rf_gs_best = GridSearchCV(rf,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(x, y)
"""
{'max_depth': 2,
 'max_features': 2,
 'min_samples_split': 3,
 'n_estimators': 150}
"""


rf_gs_best.best_params_
rf_gs_best.best_score_

# rf_final = rf.set_params(**rf_gs_best.best_params_).fit(x, y)
rf_final = rf.set_params(max_depth= None,
 max_features=None,
 min_samples_split= 3,
 n_estimators=105,
bootstrap=True,
).fit(x, y)

rmse = np.mean(np.sqrt(-cross_val_score(rf_final,x,y,cv=5,scoring="neg_mean_squared_error")))
y_pred = rf_final.predict(x)



print(rmse)
# 8860.61816564378
rf_final.score(x,y)
# 0.9547644180325335


# graph of predictions and real values

fig = plt.figure(figsize=(10, 6))
plt.title(f"Real Values & Predicted Values with RandomForest")
plt.scatter(range(x.shape[0]), y, color='black', marker='.',label='Real Values')
plt.scatter(range(x.shape[0]), y_pred, marker='+', color='green',label='Predicted Values')
plt.legend(loc=1, prop={'size': 8})
plt.show()

# Sample
print(f"Sample Real Car Value     : {y.iloc[142][0]}\nSample Predicted Car Value: {round(y_pred[142:143][0])}")


# ------------------------------ Feature Importance ------------------------------------------
fig, axs = plt.subplots(figsize=(12,10))
rf_feature_imp = pd.DataFrame({"Value":rf_final.feature_importances_, "Feature":x.columns})
sns.barplot(x="Value",y="Feature",data=rf_feature_imp.sort_values(by="Value",ascending=False)[0:len(x)])
set_title('Feature Importance for Random Forest')
plt.tight_layout()


# ------------------------------------------ transfrom BOXCOX ----------------------------------------
"""
# degerler pozitif olmak zorundadır
# for PRICE transformation log

box_price, lam1 = stats.boxcox(df["PRICE"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(ax=ax1, data=df["PRICE"])
ax1.set_title("Ham Veri")

sns.histplot(ax=ax2, data=box_price)
ax2.set_title("Log Dönüşümü")

plt.show()

before = stats.skew(df["PRICE"])
after = stats.skew(box_price)

print(before, after)

stats.probplot(df["PRICE"], plot=plt)
stats.probplot(box_price, plot=plt)

# for PRICE transformation log

box_mileage, lam2 = stats.boxcox(df["MILEAGE"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(ax=ax1, data=df["MILEAGE"])
ax1.set_title("Ham Veri")

sns.histplot(ax=ax2, data=box_mileage)
ax2.set_title("Log Dönüşümü")

plt.show()


before = stats.skew(df["MILEAGE"])
after = stats.skew(box_mileage)

print(before, after)

stats.probplot(df["MILEAGE"], plot=plt)
stats.probplot(box_mileage, plot=plt)

"""

# ------------------------------------------ transfrom LOG ----------------------------------------

"""

from scipy import stats

stats.skew(df["MILEAGE"])
stats.skew(df["PRICE"])

# for PRICE transformation log

trans_price = np.log1p(df["PRICE"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(ax=ax1, data=df["PRICE"])
ax1.set_title("Ham Veri")

sns.histplot(ax=ax2, data=trans_price)
ax2.set_title("Log Dönüşümü")

plt.show()

before = stats.skew(df["PRICE"])
after = stats.skew(trans_price)

print(before, after)

stats.probplot(df["PRICE"], plot=plt)
stats.probplot(trans_price, plot=plt)

# for PRICE transformation log

trans_mileage = np.log1p(df["MILEAGE"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(ax=ax1, data=df["MILEAGE"])
ax1.set_title("Ham Veri")

sns.histplot(ax=ax2, data=trans_mileage)
ax2.set_title("Log Dönüşümü")

plt.show()


before = stats.skew(df["MILEAGE"])
after = stats.skew(trans_mileage)

print(before, after)
df.head()
"""