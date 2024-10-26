from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


# Dataset Analysis

cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data , columns = cancer.feature_names)
df["target"] = cancer.target


# Train Test Splitting

X =  cancer.data
y = cancer.target
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3 , random_state= 42)


# Feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # transform learned parameters from x_train to scale x_test


# Training KNN Model 

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)


# Evaluation 

y_pred = knn.predict(X_test)


# Comparing Predicted y With Actual y Values

accuracy = accuracy_score(y_test,y_pred) 
print(f"accuracy : {accuracy}")


conf_matrix = confusion_matrix(y_test , y_pred)
print(f"confusion matrix : \n {conf_matrix}")


# Hyperparameter tuning 

"""
hyperparameters in KNN : k 
    k : 1 , 2 ,3 , ... , N
    Accuracy : %A , %B , %C , ...

"""
accuracy_values = []
k_values = []

for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train , y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred) 
    accuracy_values.append(accuracy)
    k_values.append(k)


plt.figure()
plt.plot(k_values,accuracy_values, marker = "X" )
plt.title("accuracy for k values") 
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()



"""
it can be said the the k = 9 , 10 , 11 gives us the best accuracy
therefore we set the n_neighbors parameter to 9 . 
"""










