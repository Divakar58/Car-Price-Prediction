from flask import Flask,render_template,request
import pickle
#from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
import os

app=Flask(__name__)
#CORS(app)

load_dotenv()

debug=os.getenv('DEBUG')

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    if request.method=='POST':
        present_price=float(request.form['showroom_price'])
        kmtravelled=float(request.form['kmtravelled'])
        #year=int(request.form['year'])
        noyear=int(request.form['noyear'])
        fueltype=request.form['fueltype']
        if(fueltype=='Diesel'):
            diesel,petrol=1,0
        elif(fueltype=='Petrol'):
            diesel,petrol=0,1
        else:
            diesel,petrol=0,0
        sellertype=int(request.form['sellertype'])
        transmission=int(request.form['transmission'])
        owner=int(request.form['noowner'])
        print([present_price,kmtravelled,sellertype,transmission,owner,diesel,petrol,noyear])
        predictedprice= model.predict(pd.DataFrame([present_price,kmtravelled,sellertype,transmission,owner,diesel,petrol,noyear],['Present_Price', 'Kms_Driven', 'Seller_Type', 'Transmission', 'Owner',
       'Diesel', 'Petrol', 'No.years']).T)
    return render_template('results.html',price=predictedprice*100000)

app.run(port=5002,debug=False)