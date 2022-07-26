from flask import Flask,request,render_template
import numpy as np
import pickle


with open('model.pkl','rb') as file:
    mymodel =pickle.load(file)

app =Flask(__name__)
@app.route('/')

def index():
    return render_template('index.html')

@app.route("/predict" ,methods = ['GET','POST'])

def predict():
    specie=''
    SepalLength = (request.form['SepalLengthCm'])
    SepalWidth = (request.form['SepalWidthCm'])
    PetalWidth = (request.form['PetalWidthCm'])
    PetalLength= (request.form['PetalLengthCm'])
    output=mymodel.predict([[SepalLength, SepalWidth, PetalWidth,PetalLength]])


    # if output==0:
    #     specie='Iris-setosa'
    # elif output==1:
    #     specie='Iris-versicolor'
    # elif output==2:
    #     specie='Iris-virginica'
    # print(specie)
    

    return render_template('index.html',predict=output[0])

if __name__=='__main__':
    app.run(debug=True)