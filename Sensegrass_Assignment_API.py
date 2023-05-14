#!/usr/bin/env python
# coding: utf-8

# # API Creation

# In[5]:


from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('wine_classifier_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming request
    features = [data['country'], data['review_description'], data['designation'],
                data['points'], data['price'], data['province'], data['region_1'],
                data['region_2'], data['winery']]

    # Make a prediction using the loaded model
    predicted_variety = model.predict([features])[0]

    # Create a response dictionary
    response = {
        'predicted_variety': predicted_variety
    }

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()


# In[ ]:




