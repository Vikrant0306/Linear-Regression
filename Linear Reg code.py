import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
car_dataset = pd.read_csv('/content/car dataset.csv')
car_dataset.head()
car_dataset.shape
car_dataset.info()
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Transmission.value_counts())
print(car_dataset.Engine.value_counts())
print(car_dataset.Seats.value_counts())
print(car_dataset.Owner_Type.value_counts())
car_dataset.replace({'Fuel_Type':{"Petrol":0,"Diesel":1,"CNG":2,"LPG":3,"Electric":4}},inplace=True)
car_dataset.replace({'Transmission':{"Manual":0,"Automatic":1}},inplace=True)
car_dataset.replace({'Owner_Type':{"First":0,"Second":1,"Third":2,"Fourth & Above":3}},inplace=True)
car_dataset.head()
x = car_dataset.drop(['Name', 'Location', 'Mileage', 'Engine', 'Power',"Seats"], axis=1)
y = car_dataset['Price']
print(x)
print(y)
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=20, random_state=2)

linear_reg=LinearRegression()

linear_reg.fit(x_train, y_train)

train_data = linear_reg.predict(x_train)

error_score = metrics.r2_score(y_train,train_data)
print('Error Score',error_score)

plt.scatter(y_train, train_data)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted prices")
plt.show()

test_data= linear_reg.predict(x_test)

error_score = metrics.r2_score(y_test,test_data)
print("Error Score", error_score)

plt.scatter(y_test, test_data)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted prices")
plt.show()
