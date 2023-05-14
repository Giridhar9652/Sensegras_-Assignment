#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd

df = pd.read_csv('OSX_DS_assignment.csv')

# Displaying the first few rows of the dataset
print(df.head())


# In[27]:


df.info()


# # Insight 1

# In[28]:


import matplotlib.pyplot as plt


# Insight 1: Top 5 countries with the most wine reviews
top_countries = df['country'].value_counts().head(5)
print("Top 5 countries with the most wine reviews:")
print(top_countries)

# Plot the top countries
plt.figure(figsize=(8, 5))
top_countries.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of Reviews')
plt.title('Top 5 Countries with the Most Wine Reviews')
plt.show()


# # Insight 2

# In[29]:


# Insight 2: Average rating by country
average_rating_by_country = df.groupby('country')['points'].mean().sort_values(ascending=False)
print("\nAverage rating by country:")
print(average_rating_by_country)

# Ploting the average rating by country
plt.figure(figsize=(10, 6))
average_rating_by_country.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Average Rating')
plt.title('Average Rating by Country')
plt.show()


# # Insight 3

# In[30]:


# Insight 3: Most common wine varieties
common_varieties = df['variety'].value_counts().head(5)
print("\nMost common wine varieties:")
print(common_varieties)

# Ploting the most common wine varieties
plt.figure(figsize=(8, 5))
common_varieties.plot(kind='bar')
plt.xlabel('Variety')
plt.ylabel('Number of Reviews')
plt.title('Most Common Wine Varieties')
plt.show()



# # Insight 4

# In[31]:


# Insight 4: Price distribution by wine variety
price_distribution = df.groupby('variety')['price'].describe()
print("\nPrice distribution by wine variety:")
print(price_distribution)


# # Insight 5

# In[32]:


# Insight 5: Correlation between points and price
correlation = df['points'].corr(df['price'])
print("\nCorrelation between points and price:")
print(correlation)

# Scatter plot of points vs. price
plt.figure(figsize=(8, 5))
plt.scatter(df['points'], df['price'])
plt.xlabel('Points')
plt.ylabel('Price')
plt.title('Correlation between Points and Price')
plt.show()


# In[33]:


import matplotlib.pyplot as plt

# Ploting the distribution of wine prices
plt.hist(df['price'], bins=20)
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Distribution of Wine Prices')
plt.show()


# In[34]:


top_varieties_rating = df.groupby('variety')['points'].mean().nlargest(5)
print("\ntop 5 varieties with the highest average rating:")
print(top_varieties_rating)
# Ploting the top 5 varieties with the highest average rating
plt.bar(top_varieties_rating.index, top_varieties_rating.values)
plt.xlabel('Variety')
plt.ylabel('Average Rating')
plt.title('Top 5 Varieties with the Highest Average Rating')
plt.xticks(rotation=45)
plt.show()


# In[35]:


average_rating_by_province = df.groupby('province')['points'].mean().sort_values(ascending=False).head(10)
print("\naverage rating by province:")
print(average_rating_by_province)
# Ploting the average rating by province
plt.barh(average_rating_by_province.index, average_rating_by_province.values)
plt.xlabel('Average Rating')
plt.ylabel('Province')
plt.title('Average Rating by Province')
plt.show()


# In[36]:


# Ploting the distribution of wine ratings
plt.hist(df['points'], bins=20)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Wine Ratings')
plt.show()


# In[37]:


top_wineries_varieties = df['winery'].value_counts().nlargest(10)
print("\ntop 10 wineries with the most wine varieties:")
print(top_wineries_varieties)
# Ploting the top 10 wineries with the most wine varieties
plt.bar(top_wineries_varieties.index, top_wineries_varieties.values)
plt.xlabel('Winery')
plt.ylabel('Count')
plt.title('Top 10 Wineries with the Most Wine Varieties')
plt.xticks(rotation=45)
plt.show()


# # Model is Random Forest Classifier

# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[39]:


features = ['country', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'winery']
target = 'variety'


# In[40]:


X = df[features]
y = df[target]


# In[41]:


# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)


# In[42]:


# Perform label encoding for categorical features
encoder = LabelEncoder()
X['country'] = encoder.fit_transform(X['country'])
X['province'] = encoder.fit_transform(X['province'])
X['region_1'] = encoder.fit_transform(X['region_1'])
X['region_2'] = encoder.fit_transform(X['region_2'])
X['winery'] = encoder.fit_transform(X['winery'])
X['designation'] = encoder.fit_transform(X['designation'])


# In[43]:


# Spliting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# Initializing the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[45]:


# Training the model
rf_classifier.fit(X_train, y_train)


# In[46]:


# Making predictions on the test set
y_pred = rf_classifier.predict(X_test)


# In[47]:


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Random Forest Classifier:", accuracy)


# # Saving The model in .pkl 

# In[48]:


import pickle

# Saving the model to a file
with open('wine_classifier_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




