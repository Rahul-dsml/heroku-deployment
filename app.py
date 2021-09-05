#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, render_template


# ### Loading the model

# In[2]:


import pickle
model= pickle.load(open('model.pkl', mode='rb'))
model


# In[4]:


import pickle

app= Flask(__name__)
model= pickle.load(open('model.pkl', mode='rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    inp= [float(x) for x in request.form.values()]
    inp= np.array(inp).reshape(-1,1)
    pred= model.predict(inp)
    out= round(pred[0], 2)
    
    return render_template('index.html', prediction_text= 'The Student is likely to get {} marks'.format(out))

if __name__=='__main__':
    app.run(debug=True)
    


# In[ ]:




