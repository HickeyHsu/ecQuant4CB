from alphanet import AlphaNetV3, load_model
from alphanet.data import TrainValData, TimeSeriesData
from alphanet.metrics import UpDownAccuracy
import pandas as pd
import numpy as np
import tensorflow as tf
import sys,os,pickle,copy
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
os.chdir(os.path.split(os.path.abspath(__file__))[0])
from bt_utils import standard,MaxDrawdown
class AlphaCB:
    def __init__(self,data_path='./data210917b.pkl',begin:int=0):
        self.begin_index=begin#从第几日开始——涉及一些需要历史数据的指标
        self.data_load(data_path)
        self.bond_metric_DF=self.get_bond_metric_DF()

    def data_load(self,data_path):
        data_file = open(data_path, 'rb')
        self.allData=pickle.load(data_file).reset_index(drop=True)
        # 只要能成功构造包含date，code,price和所需因子的dataframe:raw就能进行回测
        raw=pd.DataFrame()
        raw['date']=self.allData['base_date']
        raw['code']=self.allData['转债代码']
        raw['price']=self.allData['转债价格']
        raw['remain']=self.allData['转债余额']
        raw['convert_premium_ratio']=self.allData['转股溢价率'].apply(lambda x:float(x.strip("%")))
        raw['cb_rt']=self.allData['涨跌'].apply(lambda x:float(x.strip("%")))
        raw['st_rt']=self.allData['涨跌.1'].apply(lambda x:float(x.strip("%")))
        raw['zfdb']=(raw['cb_rt']-raw['st_rt'])/(100+raw['convert_premium_ratio'])
        raw['turn_rt']=self.allData['转债换手率'].apply(lambda x:float(x.strip("%")))
        raw['avg_cpr']=self.allData['avg_cpr']
        raw['dev']=raw['avg_cpr']-raw['convert_premium_ratio']
        raw['cv']=self.allData['cv']
        raw['dblow']=raw['price']+raw['convert_premium_ratio']
        # raw['cp_BE']=self.allData['cp_BE']
        # raw['cpr_BE']=raw['cp_BE']-raw['price']#光大法修正的溢价率
        raw['st_trend']=self.allData['st_trend'].astype(int)
        raw['ma20']=self.allData['MA20乖离'].apply(lambda x:float(x.strip("%")))
        raw['slope']=self.allData['slope']
        self.raw=raw.iloc[self.begin_index:,:]

    def compute_future_return(self,n:int=5):
        grouped=self.bond_metric_DF.groupby('code')
        df_list=[]
        for group_name, group_df in grouped:
            group_df=group_df.sort_values('date', ascending=True,ignore_index=True)
            group_df['future_return']=group_df['price'].pct_change(periods=n).shift(-n)
            group_df=group_df.iloc[:-n]
            df_list.append(group_df)
        result=pd.concat(df_list,ignore_index=True)
        return result
    def get_bond_metric_DF(self) ->pd.DataFrame:
        rawdf=self.raw
        metrics2mad=['convert_premium_ratio','zfdb','remain','price','dev','cv','dblow','st_trend','ma20','slope']#这里是要用到的因子
        bond_metric_DF=standard(metrics2mad,rawdf)
        return bond_metric_DF



def test(dotrain=False):
    anet =AlphaCB()
    # compute label (future return)
    df=anet.compute_future_return(10)
    df['int_date']=df["date"].apply(lambda x:int(x.strftime('%Y%m%d')))


    # create an empty list
    stock_data_list = []
    # put each stock into the list using TimeSeriesData() class
    # codes = df["code"].unique()

    test_df=df[df.code==128144]
    # for code in codes:
    #     table_part = df.loc[df["code"] == code, :]
    #     print(table_part)
    #     break
    grouped=df.groupby('code')
    # metrics_std=['convert_premium_ratio_std','zfdb_std','remain_std',
    #     'price_std','dev_std','cv_std','ma20_std','slope_std'
    # ]
    metrics_std=['convert_premium_ratio','zfdb','remain',
        'price','dev','cv','ma20','slope'
    ]
    for group_name, group_df in grouped:
        stock_data_list.append(TimeSeriesData(
            dates=group_df["int_date"].values,                   # date column            
            data=group_df[metrics_std].values,                # data columns
            labels=group_df["future_return"].values # label column
        ))

    # put stock list into TrainValData() class, specify dataset lengths
    train_val_data = TrainValData(time_series_list=stock_data_list,
        train_length=300,   # 1200 trading days for training
        validate_length=100, # 150 trading days for validation
        history_length=20,   # each input contains 30 days of history
        sample_step=1,       # jump to days forward for each sampling
        train_val_gap=10     # leave a 10-day gap between training and validation
    )
    
    # get one training period that start from 
    train, val, dates_info = train_val_data.get(20190918, order="by_date")
    # test_set,test_dateinfo=train_val_data.get
    # print(dates_info)
    if dotrain:
        # get an AlphaNetV3 instance
        model = AlphaNetV3(l2=0.001, dropout=0.0)

        # you may use UpDownAccuracy() here to evaluate performance
        model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError(),
                            UpDownAccuracy()]
        )
        # train
        model.fit(train.batch(500).cache(),
                validation_data=val.batch(500).cache(),
                epochs=300)

        # save model by save method
        model.save("model.bt")

        # or just save weights
        model.save_weights("weights.bt")

    
    # load entire model using load_model() from alphanet module
    model = load_model("model.bt")

    # only load weights by first creating a model instance
    model = AlphaNetV3(l2=0.001, dropout=0.0)
    model.load_weights("weights.bt")

    print(val.batch(500).cache())
    # test_data=tf.convert_to_tensor()
    # test_data=test_df[metrics_std].values[:20,:]
    # print(test_data.shape)
    output=model.predict(val.batch(500).cache())
    print(output)
if __name__ == '__main__':
    test()
