from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from datetime import date
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from math import sqrt
from pyspark.sql.functions import rand
import sys

def convert_time_to_quarter(string):
    hrs = int(string.split(":")[0])
    return int(hrs / 6)

import warnings
warnings.filterwarnings("ignore")

def map1(line):
    new_features = []

    #Create and append features one by one

    vendor_id = int(line[0]) - 1                    #Vendor_id is either 1 or 2
    new_features.append(vendor_id)

    date_time = line[1]                             #Pickup DateTime
    date1, time = date_time.split(" ")
    y, m, d = date1.split("-")
    day = date(int(y), int(m), int(d)).weekday()
        
    for i in range(7):                              #Day of the week is extracted from the date
        if i == day:
            new_features.append(1)
        else:
            new_features.append(0)
            
    for i in range(4):                              #Quarter of the day is extracted using the hours
        if i == convert_time_to_quarter(time):
            new_features.append(1)
        else:
            new_features.append(0)
            
    for i in range(6):                              #Month number (Jan-June) is extracted from date
        if int(m) == i+1:
            new_features.append(1)
        else:
            new_features.append(0)

    #We can use dropoff datetime to extract features as well but in the kaggle test dataset it is missing
    #and these features have a high correlation with the ones generated using pickup datetime
    """
    feature = line[2]
    date1, time = date_time.split(" ")
    y, m, d = date1.split("-")
    day = date(int(y), int(m), int(d)).weekday()
        
    for i in range(7):                              #Day of the week is extracted from the date
        if i == day:
            new_features.append(1)
        else:
            new_features.append(0)
            
    for i in range(4):                              #Quarter of the day is extracted using the hours
        if i == convert_time_to_quarter(time):
            new_features.append(1)
        else:
            new_features.append(0)
            
    for i in range(6):                              #Month number (Jan-June) is extracted from date
        if int(m) == i+1:
            new_features.append(1)
        else:
            new_features.append(0)
    """

    num_pass = int(line[3])
    new_features.append(num_pass)

    pick_up_lon = float(line[4])
    pick_up_lat = float(line[5])

    drop_off_lon = float(line[6])
    drop_off_lat = float(line[7])

    lat_dist = drop_off_lat - pick_up_lat
    new_features.append(lat_dist)

    lon_dist = drop_off_lon - pick_up_lon
    new_features.append(lon_dist)

    manhattan_dist = abs(lat_dist) + abs(lon_dist)
    new_features.append(manhattan_dist)

    #save information
    if line[8] == "N":
        new_features.append(0)
    else:
        new_features.append(1)

    trip_duration = int(line[9])
    new_features.append(trip_duration)

    return new_features

def parse_data(line):
    values = [float(x) for x in line]
    return LabeledPoint(values[-1], values[:-1])

conf = SparkConf().setMaster("local[*]").setAppName("Regression_GD")
sc = SparkContext(conf=conf)

spark = SparkSession(sc)
sqlContext = SQLContext(sc)
text_file = sc.textFile("train.csv")
out1 = text_file.map(lambda line: line.split(","))
portion = float(sys.argv[1])*float(out1.count())/100.0

#removes first tuple (column names) and first column(trip id)
out2 = out1.zipWithIndex().filter(lambda tup: tup[1]>1 and tup[1]<portion).map(lambda x:x[0]).map(lambda a: a[1:])
total = out2.count()

#creates training set and shuffles it
out_train = out2.zipWithIndex().filter(lambda tup: tup[1] < int(0.8*total)).map(lambda x:x[0]).toDF().orderBy(rand()).rdd

#creates test set
out_test = out2.zipWithIndex().filter(lambda tup: tup[1] > int(0.8*total)).map(lambda x:x[0])

parsed_train_data = out_train.map(map1).map(parse_data)
parsed_test_data = out_test.map(map1).map(parse_data)
TrainSize = int(total*0.8)

#Code used to perform 10-fold Cross Validation to find best parameters
"""
folds = 10
TSize = float(TrainSize/folds)
total_rmse = 0
total_mae = 0
for i in range(folds):
    print("Fold "+str(i+1))
    ZippedData = parsed_train_data.zipWithIndex()
    CVTrainData = ZippedData.filter(lambda tup: tup[1]<int(TSize*i) or tup[1]>int(TSize*(i+1))).map(lambda x:x[0])
    CVTestData = ZippedData.filter(lambda tup: tup[1]>int(TSize*i) and tup[1]<int(TSize*(i+1))).map(lambda x:x[0])
    model = LinearRegressionWithSGD.train(CVTrainData, iterations=10000, step=0.01, miniBatchFraction=0.005)
    values_and_preds = CVTestData.map(lambda p: (p.label, model.predict(p.features)))
    RMSE = sqrt(values_and_preds.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y)/values_and_preds.count())
    total_rmse += RMSE
    MAE = values_and_preds.map(lambda vp: abs(vp[0] - vp[1])).reduce(lambda x, y: x + y)/values_and_preds.count()
    total_mae += MAE
    print(RMSE)
    print(MAE)

print("Avg Root Mean Squared Error on CV = " + str(total_rmse/folds))
print("Avg Mean Absolute Error on CV = " + str(total_mae/folds))
"""

test_model = LinearRegressionWithSGD.train(parsed_train_data,iterations=10000, step=0.01, miniBatchFraction=0.005)
values_and_preds = parsed_test_data.map(lambda p: (p.label, test_model.predict(p.features)))
TestRMSE = sqrt(values_and_preds.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y)/values_and_preds.count())
print("Root Mean Squared Error on Test Data = " + str(TestRMSE))
TestMAE = values_and_preds.map(lambda vp: abs(vp[0] - vp[1])).reduce(lambda x, y: x + y)/values_and_preds.count()
print("TMean Absolute Error on Test Data = " + str(TestMAE))
