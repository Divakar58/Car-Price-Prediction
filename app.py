from flask import Flask,render_template,request
import pickle
#from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
import os
import locale

locale.setlocale(locale.LC_MONETARY,'en_IN')

app=Flask(__name__)
#CORS(app)

load_dotenv()


DEBUG=eval(os.environ.get("DEBUG"))

PRODUCTION=eval(os.environ.get("PRODUCTION"))


PORT = int(os.environ.get("PORT"))



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
        predictedprice= model.predict(pd.DataFrame([present_price,kmtravelled,sellertype,transmission,owner,diesel,petrol,noyear],['Present_Price', 'Kms_Driven', 'Seller_Type', 'Transmission', 'Owner',
        'Diesel', 'Petrol', 'No.years']).T)
        predictedprice=round(predictedprice[0]*100000,2)
    return render_template('results.html',price='\u20B9' +" "+ locale.currency(predictedprice,symbol=False, grouping=True))

if __name__ == "__main__":
    if(PRODUCTION):
        app.run()   
    else:
       print("app running at port",PORT,"debug mode",DEBUG)
       app.run(port=PORT,debug=DEBUG)