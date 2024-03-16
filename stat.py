import requests
from datetime import datetime,time,timedelta
import numpy as np

def error(W,x,b,y):
    #w->n*n
    #b->n*4
    #x->n*4
    #y->n*4
    ypred = np.matmul(W,x) + b
    E = ypred - y
    E = np.square(E)
    return np.sum(np.sum(E))


def gradW(W,x,b,y):
    ypred = np.matmul(W,x) + b
    gradE = y - ypred
    return -2*np.matmul(gradE,(x.T))

def gradb(W,x,b,y):
    ypred = np.matmul(W,x) + b
    return -2*(y - ypred)

if __name__ == '__main__':
    date_time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    expiries = "https://live.markethound.in/api/history/expiries?index=NIFTY"
    date = ""
    dte = ""
    # data = "https://live.markethound.in/api/history/decay?name=NIFTY&expiry="+date+"&dte="+dte
    response = requests.get(expiries)
    data = response.json()
    today = "2024-03-11T00:00:00.000Z"
    # print(data['result']) #contains list of expiries
    #this code assumes that today is monday (11-03-24) and we are going to be predicting for tuesday (12-03-24)
    expiryDates = []
    for st in data['result']:
        expiryDates.append(datetime.strptime(st, date_time_format).date())
    today = datetime.strptime(today,date_time_format).date()
    # print(expiryDates)
    # print(today)
    for i in expiryDates:
        if i>today:
            expiryDates.remove(i)
    #above code gets rid of dates on which we do not have any data
    #now to get data..
    
    #we want to focus on influence of 3dte prices on 2dte prices (since our day is 3dte and we are predicting 2dte)
    inputData = {}
    outputData = {}
    
    feasibleDates = []

    for date in expiryDates:
        #first get 3dte data
        dte = 3
        start_time = time(hour=14, minute=0)
        end_time = time(hour=15, minute=0)

        # Initialize current time to start time
        current_time = start_time

        # Define the time increment (1 minute)
        time_increment = timedelta(minutes=15)

        thing = []
        while current_time <= end_time: # there is no point whatsoever to doing this. The data is the exact same for all times :(
            t = np.zeros((4,1))
            dateTime = datetime.combine(date,current_time)
            current_time = (datetime.combine(datetime.min, current_time) + time_increment).time()
        # print(date.strftime(date_time_format))
            dte = str(dte)
            dataUrl = "https://live.markethound.in/api/history/decay?name=NIFTY&expiry="+dateTime.strftime(date_time_format)+"&dte="+dte
            obtained = requests.get(dataUrl)
            if obtained.status_code != 200:
                break
            obtained = obtained.json()
            obtained = obtained['result']
            # print(obtained)
            # exit(0)
            if len(obtained) == 0:
                break
            #obtained contains a ton of stuff we do not care about, we only care about open,close,high and low
            # thing[current_time] = obtained
            #we only care about obtained[0] since the rest are already included
            # useful = []
            # useful.append(obtained[0]['open'])
            # useful.append(obtained[0]['close'])
            # useful.append(obtained[0]['high'])
            # useful.append(obtained[0]['low'])
            t[0] = obtained[0]['open']
            t[1] = obtained[0]['close']
            t[2] = obtained[0]['high']
            t[3] = obtained[0]['low']
            # t[4] = obtained[0]['intradayMovement']
            thing.append(t.T)
            # print(useful,date)
            # print(obtained)
        if len(thing) != 0:
            matr = np.vstack(thing)
            # print(matr)
            # exit(0)
            feasibleDates.append(date)
            inputData[date] = matr

        dte = 2
        start_time = time(hour=14, minute=0)
        end_time = time(hour=15, minute=0)

        # Initialize current time to start time
        current_time = start_time

        # Define the time increment (1 minute)
        time_increment = timedelta(minutes=15)

        thing = []
        t = np.zeros((4,1))
        while current_time <= end_time:
            dateTime = datetime.combine(date,current_time)
            current_time = (datetime.combine(datetime.min, current_time) + time_increment).time()
        # print(date.strftime(date_time_format))
            dte = str(dte)
            dataUrl = "https://live.markethound.in/api/history/decay?name=NIFTY&expiry="+dateTime.strftime(date_time_format)+"&dte="+dte
            obtained = requests.get(dataUrl)
            if obtained.status_code != 200:
                break
            obtained = obtained.json()
            obtained = obtained['result']
            if len(obtained) == 0:
                break
            t[0] = obtained[0]['open']
            t[1] = obtained[0]['close']
            t[2] = obtained[0]['high']
            t[3] = obtained[0]['low']
            thing.append(t.T)
            # print(obtained)
        if len(thing) != 0:
            matr = np.vstack(thing)
            outputData[date] = matr
    # print(inputData)
    #input and output data now have the data... now we need to train a model
    #i am just going to use simple linear regression with an L2 norm to prevent overfitting
    n = len(outputData[feasibleDates[0]])
    W = np.zeros((n,n))
    b = np.zeros((n,4))

    counter = 0
    lr = 0.00000001
    epsilon = 1

    prevErr = 1000000000000
    err = 0

    #we will use stochastic gradient descent
    while abs(prevErr - err) > epsilon: #to prevent overfitting
        # index = np.random.randint(0,len(feasibleDates))
        E = 0
        for index in range(len(feasibleDates)):
            E += error(W,inputData[expiryDates[index]],b,outputData[expiryDates[index]])
            W -= lr*gradW(W,inputData[expiryDates[index]],b,outputData[expiryDates[index]])
            b -= lr*gradb(W,inputData[expiryDates[index]],b,outputData[expiryDates[index]])
        prevErr = err
        err = E
        print(E,prevErr,counter)
        counter+=1
    
    #now prediction
    x = []
    # todayDate = "2024-"
    dataUrl = "https://live.markethound.in/api/history/decay?name=NIFTY&expiry=2024-03-14T00:00:00.000Z&dte=3" #change this in case your input changes per minute
    obtained = requests.get(dataUrl)
    obtained = obtained.json()
    # print(obtained)
    # exit(0)
    obtained = obtained['result']
    start_time = time(hour=14, minute=0)
    end_time = time(hour=15, minute=0)

    # Initialize current time to start time
    current_time = start_time

    # Define the time increment (1 minute)
    time_increment = timedelta(minutes=15)

    thing = []
    while current_time <= end_time: # there is no point whatsoever to doing this. The data is the exact same for all times :(
        t = np.zeros((4,1))
        t[0] = obtained[0]['open']
        t[1] = obtained[0]['close']
        t[2] = obtained[0]['high']
        t[3] = obtained[0]['low']
        current_time = (datetime.combine(datetime.min, current_time) + time_increment).time()
        # t[4] = obtained[0]['intradayMovement']
        thing.append(t.T)
        # print(useful,date)
        # print(obtained)
    if len(thing) != 0:
        x = np.vstack(thing)
    yPred = np.matmul(W,x) + b
    print(yPred)
    
    y = []

    dataUrl = "https://live.markethound.in/api/history/decay?name=NIFTY&expiry=2024-03-14T00:00:00.000Z&dte=2"
    obtained = requests.get(dataUrl)
    obtained = obtained.json()
    # print(obtained)
    # exit(0)
    obtained = obtained['result']
    start_time = time(hour=14, minute=0)
    end_time = time(hour=15, minute=0)

    # Initialize current time to start time
    current_time = start_time

    # Define the time increment (1 minute)
    time_increment = timedelta(minutes=15)

    thing = []
    while current_time <= end_time: # there is no point whatsoever to doing this. The data is the exact same for all times :(
        t = np.zeros((4,1))
        t[0] = obtained[0]['open']
        t[1] = obtained[0]['close']
        t[2] = obtained[0]['high']
        t[3] = obtained[0]['low']
        current_time = (datetime.combine(datetime.min, current_time) + time_increment).time()
        # t[4] = obtained[0]['intradayMovement']
        thing.append(t.T)
        # print(useful,date)
        # print(obtained)
    if len(thing) != 0:
        y = np.vstack(thing)
    print(y)