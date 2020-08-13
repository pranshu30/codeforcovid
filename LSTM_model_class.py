import numpy as np
import pandas as pd
from numpy import hstack
from numpy import array
import os, sys
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import warnings
import argparse
warnings.filterwarnings("ignore")


features_list=['is_stayathome',
 'retail',
 'grocery',
 'parks',
 'transit',
 'workplace',
 'residential',
 'temp',
"daily_cases_rollingmean_3",
"daily_cases_rollingmean_5",
 'retail_rollingmean_5',
 'retail_rollingmean_9',
 'retail_rollingmean_11',
 'retail_rollingmean_14',
 'grocery_rollingmean_5',
 'grocery_rollingmean_9',
 'grocery_rollingmean_11',
 'grocery_rollingmean_14',
 'parks_rollingmean_5',
 'parks_rollingmean_9',
 'parks_rollingmean_11',
 'parks_rollingmean_14',
 'transit_rollingmean_5',
 'transit_rollingmean_9',
 'transit_rollingmean_11',
 'transit_rollingmean_14',
 'workplace_rollingmean_5',
 'workplace_rollingmean_9',
 'workplace_rollingmean_11',
 'workplace_rollingmean_14',
 'residential_rollingmean_5',
 'residential_rollingmean_9',
 'residential_rollingmean_11',
 'residential_rollingmean_14',
 'temp_rollingmean_5',
 'temp_rollingmean_9',
 'temp_rollingmean_11',
 'temp_rollingmean_14']

class casePrediction:
    def __init__(self,input_folder, model_output_dir,n_steps,n_predict_steps,features_list,
                 train_epochs=220,split_into_train_test=False, split_index=154, output_errors_csv=False,
                 save_model_h5_after_train=True
                ):
        self.input_folder=input_folder
        self.model_output_dir=model_output_dir
        self.csv_files=os.listdir(input_folder)
        self.n_steps=n_steps
        self.split_into_train_test=split_into_train_test
        self.split_index=split_index
        self.train_epochs=train_epochs
        self.n_predict_steps=n_predict_steps
        self.features_list=features_list
        self.save_model_h5_after_train=save_model_h5_after_train
        self.output_errors_csv=output_errors_csv
        self.county_names_log=[]
        self.actuals_log=[]
        self.predictions_log=[]
        self.MAE_log=[]
        self.MSE_log=[]
        self.RMSE_log=[]
        self.CVRMSE_log=[]
        self.five_day_err_log=[]
        self.five_day_actual_total_log=[]
        self.five_day_pcterr_log=[]
        self.five_day_pred_total_log=[]
        self.counter=0
    
    def build_all_models(self):
        for csv_file in self.csv_files:
            fp=os.path.join(self.input_folder,csv_file)
            self.read_test_csv(fp)
            self.process_dataset()
            self.make_input_arrs()
            #self.scale_daily_cases()
            self.train_model()
            self.test_model()
            if self.save_model_h5_after_train:
                self.save_model_h5()
            if self.output_errors_csv and self.counter%10==0:
                self.output_errors_df()
        if self.output_errors_csv:
            self.output_errors_df()
    
    def build_model_for_county(self, county_name):
        csv_filename="{}_data.csv".format(county_name.lower().title())
        fp=os.path.join(self.input_folder,csv_filename)
        self.read_test_csv(fp)
        self.process_dataset()
        self.make_input_arrs()
        self.train_model()
        if self.save_model_h5_after_train:
            self.save_model_h5()
    
    def read_test_csv(self, csv_fp):
        self.csv_df=pd.read_csv(csv_fp)
        self.county_name=self.csv_df.iloc[0]['state_county']
        self.features=self.csv_df[self.features_list]
        
    def process_dataset(self):
        self.features_diff_df=self.features.copy(deep=True)
        self.features=self.features_diff_df.values
        self.daily_cases_diff_df=self.csv_df[['daily_cases']]
        self.daily_cases_diff=self.daily_cases_diff_df.values
        
    def make_input_arrs(self):
        x_arr=[]
        y_arr=[]
        for i in range(len(self.features)): 
            if i<self.n_steps:               
                continue
            if not self.split_into_train_test and i==len(self.features)-self.n_predict_steps:
                break
            elif  self.split_into_train_test and i==len(self.features):
                break
            start_index=i-self.n_steps          
            seq_x = self.features[start_index:i]  
            x_arr.append(seq_x)
            new_y=[[x] for x in self.daily_cases_diff_df['daily_cases'].iloc[i:i+self.n_predict_steps].values.tolist()]
            new_y=new_y+[[0]]*(self.n_predict_steps-len(new_y))
            y_arr.append(new_y)
        
        if self.split_into_train_test:
            self.x_train=array(x_arr[:self.split_index])
            self.y_train=array(y_arr[:self.split_index])
            self.x_test=x_arr[self.split_index:]
            self.y_test=array(y_arr[self.split_index:])
            self.n_features=self.x_train.shape[2]
        else:
            self.x_train=array(x_arr)
            self.y_train=array(y_arr)
            self.x_test=[x_arr[-1]]
            self.y_test=array([y_arr[-1]])
            self.n_features=self.x_train.shape[2]
    
    def output_forecast(self, county_name, start_date=None): # start date in form of YYYY-MM-DD, count name in form of "California_Alameda" or "New York_New York City" 
        print('forecasting for {}'.format(county_name))
        if start_date is None:
            yesterday_date=datetime.now()-timedelta(days=1)
            start_date=yesterday_date.strftime("%Y-%m-%d")
        try:
            self.read_test_csv(os.path.join(self.input_folder, "{}_data.csv".format(county_name.lower().title()) ))
            self.process_dataset()
        except:
            print("input csv file not found")
            return -1
        
        end_row=self.csv_df.loc[self.csv_df['date'] == start_date]
        if end_row.empty:
            print("start date not found in csv")
            return -2
        end_index=end_row.index[0]
        model_h5_exists=self.check_for_saved_model(county_name.lower())
        if model_h5_exists:
            print("saved model found")
            self.model=load_model(os.path.join(self.model_output_dir,"{}.h5".format(county_name.lower()) ))
        else:
            try:
                print("saved model not found, training new")
                self.build_model_for_county(county_name)
            except:
                print("model build for county failed")
                return -3
        

        try:
            self._produce_forecast(end_index)
        except:
            print("error during forecast")
            return -4
        return self.forecast
    
    
    def _produce_forecast(self,end_index):
        start_index=end_index-self.n_steps
        self.input_x_arr= self.features[start_index:end_index]  
        yhat = self.model.predict(array([self.input_x_arr]), verbose=0)
        self.forecast=[max([x[0],0]) for x in yhat[0]] # max used to filter out negative case predictions
    
    def check_for_saved_model(self,county_name):
        h5_file="{}.h5".format(county_name.lower())
        if h5_file in os.listdir(self.model_output_dir):
            return True
        return False
    
    def train_model(self):
        print("Train for {}".format(self.county_name))
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=(self.n_steps, self.n_features)))
        self.model.add(RepeatVector(self.n_predict_steps))
        self.model.add(LSTM(200, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(array(self.x_train),  self.y_train, epochs=self.train_epochs, verbose=0)


    def save_model_h5(self): 
        h5_file="{}.h5".format(self.county_name.lower())
        self.model.save(os.path.join(self.model_output_dir,h5_file))

    
    def test_model(self):
        print("Testing")
        county_abs_errs=[]
        test_input=self.x_test[0]
        #actuals=[y[0] for y in self.y_test_scaled[0]]
        actuals=[x for  x in self.y_test[0]]
        yhat = self.model.predict(array([test_input]), verbose=0)
        pred=[x[0] for x in yhat[0]]
        county_errs=[x for x in np.subtract(actuals,pred)]
        county_abs_errs=np.abs(county_errs)
        county_sq_err=[x**2 for x in county_errs]
        MAE=np.mean(county_abs_errs)
        MSE=np.mean(county_sq_err)
        RMSE=np.sqrt(MSE)
        CVRMSE=RMSE/np.mean(pred)    
        five_day_err=np.sum(actuals)-np.sum(pred)
        self.county_names_log.append(self.county_name)
        self.actuals_log.append("|".join([str(x[0]) for x in actuals]))
        self.predictions_log.append("|".join([str(int(x)) for x in pred]))
        self.MAE_log.append(MAE)
        self.MSE_log.append(MSE)
        self.RMSE_log.append(RMSE)
        self.CVRMSE_log.append(CVRMSE)
        self.five_day_actual_total_log.append(np.sum(actuals))
        self.five_day_pred_total_log.append(np.sum(pred))
        self.five_day_err_log.append(five_day_err)
        self.five_day_pcterr_log.append(five_day_err/np.sum(actuals))
        self.counter+=1

        
    def output_errors_df(self):
        output_df=pd.DataFrame(
                {"county":self.county_names_log,
                 "actual_cases":self.actuals_log,
                 "predicted_cases":self.predictions_log,
                 "MAE":self.MAE_log,
                 "MSE":self.MSE_log,
                 "RMSE":self.RMSE_log,
                 "CVRMSE":self.CVRMSE_log,
                 "seven_day_actual_total":self.five_day_actual_total_log,
                 "seven_day_prediction_total":self.five_day_pred_total_log,
                 "seven_day_err":self.five_day_err_log,
                 "seven_day_pct_err":self.five_day_pcterr_log
                 
                }
        )
        output_df.to_csv("eval_folder/lstm_errors_log_{}.csv".format(datetime.now().strftime("%m%d_%H%M")))


if __name__=="__main__":
    print("Code 4 Covid")
    parser = argparse.ArgumentParser(description='Covid case forecasting with LSTM')
    
    parser.add_argument('mode', type=str, default="f",
                    help='a for forecast of all counties in input folder, f for seven forecast of single county, b for build all county models')
    parser.add_argument('-c', metavar="county", 
                    help='county such as California_Alameda')
    parser.add_argument('-d', metavar="forecast_from_date", 
                help='forecast from date in YYYY-MM-DD form', default=None)
    parser.add_argument('--s', metavar='model_h5_directory', help="directory where h5 files saved", 
                            default="model_h5_files")
    parser.add_argument('--i', metavar='input_data_csv_directory', help="directory of input csv files", 
                        default="county_data_roll_3579_11_13_14")
    parser.add_argument('--t', metavar='test_train', help="use test train split", type=bool,
                        default=False)
    parser.add_argument('--w', metavar='save_h5_files', help="save model h5 files", type=bool,
                        default=True)
    parser.add_argument('--e', metavar='epochs', help="number of epochs for training", type=int,
                    default=220)
    parser.add_argument('--p', metavar='output_test_errors_csv', help="output test errors csv", type=bool,
                default=False)
    args=parser.parse_args()
    if args.mode=='f' and args.c is None:
        print("No county provided")
        sys.exit()

    input_csv_folder=args.i
    model_h5_folder=args.s
    use_test_train_split=args.t
    save_model_h5=args.w
    epochs=args.e
    county=args.c
    forecast_date=args.d
    output_errors_csv=args.p

    LSTM_class=casePrediction(input_csv_folder,model_h5_folder,n_steps=8,n_predict_steps=7,features_list=features_list,
                           train_epochs=epochs, split_into_train_test=use_test_train_split, 
                           save_model_h5_after_train=save_model_h5, output_errors_csv=output_errors_csv)
    forecast_tuples=[]
    if args.mode=='f':
        seven_day_forecast=LSTM_class.output_forecast(county,forecast_date)
        forecast_tup=tuple(county,forecast_date,seven_day_forecast)
        print(forecast_tup)
    elif args.mode=='a':
        county_list=[]
        forecast_date_list=[]
        forecast_cases_list=[]
        for csv_file in os.listdir(input_csv_folder):
            county=csv_file.split("_data.csv")[0]
            seven_day_forecast=LSTM_class.output_forecast(county,forecast_date)
            if type(seven_day_forecast) == int and seven_day_forecast<0:
                print("invalid start date")
                sys.exit()
            county_list.append(county)
            forecast_date_list.append(forecast_date)
            forecast_cases_list.append("|".join([str(x) for x in seven_day_forecast]))
        forecast_df=pd.DataFrame(
            {
                'county':county_list,
                'start_date':forecast_date_list,
                'seven_day_forecast':forecast_cases_list
            }
        )            
        forecast_df.to_csv('forecasts_{}_{}.csv'.format(forecast_date,datetime.now().strftime("%m%d_%H%M")))
    elif args.mode=='b':
        LSTM_class.build_all_models()
    print("Done")