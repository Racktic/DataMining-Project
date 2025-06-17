import pandas as pd
import pickle
 
raw=pd.read_csv('train.csv')
train_raw=raw[raw['order_pay_time'] <= '2013-07-31 23:59:59']
raw.sort_values('order_pay_time', ascending=True, inplace=True)

label_raw=set(raw[raw['order_pay_time'] > '2013-07-31 23:59:59']['customer_id'].dropna())
def preprocess(raw):
    data = pd.DataFrame(
        raw.groupby('customer_id')['customer_gender'].last().fillna(0)
    )  
    # last user-good activity
    data[['goods_id_last','goods_status_last','goods_price_last','goods_has_discount_last','goods_list_time_last',
          'goods_delist_time_last']]= \
        raw.groupby('customer_id')[['goods_id', 'goods_status', 'goods_price', 'goods_has_discount', 'goods_list_time',
                               'goods_delist_time']].last()
    
    # last user-order activity
    data[['order_total_num_last','order_amount_last','order_total_payment_last','order_total_discount_last','order_pay_time_last',
          'order_status_last','order_count_last','is_customer_rate_last','order_detail_status_last', 'order_detail_goods_num_last', 
          'order_detail_amount_last','order_detail_payment_last', 'order_detail_discount_last']]= \
        raw.groupby('customer_id')[['order_total_num', 'order_amount','order_total_payment', 'order_total_discount', 'order_pay_time',
               'order_status', 'order_count', 'is_customer_rate','order_detail_status', 'order_detail_goods_num', 
                'order_detail_amount','order_detail_payment', 'order_detail_discount']].last()     
    
    
   # last user-member activity
    data[['member_id_last','member_status_last','is_member_actived_last']]= \
        raw.groupby('customer_id')[['member_id','member_status','is_member_actived']].last() 
    
    # goods_price
    data[['goods_price_min','goods_price_max','goods_price_mean','goods_price_std']]= \
        raw.groupby('customer_id',as_index = False)['goods_price'].agg([
            ('goods_price_min', 'min'),
            ('goods_price_max', 'max'),
            ('goods_price_mean', 'mean'),
            ('goods_price_std', 'std')
        ]).drop(['customer_id'],axis=1)
    
    
    # order_total_payment
    data[['order_total_payment_min','order_total_payment_max','order_total_payment_mean','order_total_payment_std']]= \
        raw.groupby('customer_id',as_index = False)['order_total_payment'].agg([
            ('order_total_payment_min', 'min'),
            ('order_total_payment_max', 'max'),
            ('order_total_payment_mean', 'mean'),
            ('order_total_payment_std', 'std')
        ]).drop(['customer_id'],axis=1)
    
    # user total order count
    data[['order_count']] = raw.groupby('customer_id',as_index = False)['order_id'].count().drop(['customer_id'],axis=1)
    
    # user total goods count
    data[['goods_count']] = raw.groupby('customer_id',as_index = False)['goods_id'].count().drop(['customer_id'],axis=1)
    
    # is_customer_rate
    data[['is_customer_rate_mean','is_customer_rate_sum']]=raw.groupby('customer_id')['is_customer_rate'].agg([
        ('is_customer_rate_mean', "mean"),
        ('is_customer_rate_sum', "sum")
    ])
    
    # order discount
    data['discount']=data['order_detail_amount_last']/data['order_detail_payment_last']
    
    # member_status
    data[['member_status_mean','member_status_sum']]=raw.groupby('customer_id')['member_status'].agg([
        ('member_status_mean', "mean"),
        ('member_status_sum', "sum")
    ])
    
    # order_detail_status
    data[['order_detail_discount_mean','order_detail_discount_sum']]=raw.groupby('customer_id')['order_detail_discount'].agg([
        ('order_detail_discount_mean', "mean"),
        ('order_detail_discount_sum', "sum")
    ])      
    
    # goods_status
    data[['goods_status_mean','goods_status_sum']]=raw.groupby('customer_id')['goods_status'].agg([
        ('goods_status_mean', "mean"),
        ('goods_status_sum', "sum")
    ])   
    
    # is_member_actived
    data[['is_member_actived_mean','is_member_actived_sum']]=raw.groupby('customer_id')['is_member_actived'].agg([
        ('is_member_actived_mean', "mean"),
        ('is_member_actived_sum', "sum")
    ])  
    
    # order_status
    data[['order_status_mean','order_status_sum']]=raw.groupby('customer_id')['order_status'].agg([
        ('order_status_mean', "mean"),
        ('order_status_sum', "sum")
    ])
    
    # goods_has_discount   
    data[['goods_has_discount_mean','goods_has_discount_sum']]= raw.groupby('customer_id')['goods_has_discount'].agg([
        ('goods_has_discount_mean', "mean"),
        ('goods_has_discount_sum', "sum")
    ])
    
    # order_total_payment
    data[['order_total_payment_mean','order_total_payment_sum']]= raw.groupby('customer_id')['order_total_payment'].agg([
        ('order_total_payment_mean', "mean"),
        ('order_total_payment_sum', "sum")
    ])
        
    # order_total_num
    data[['order_total_num_mean','order_total_num_sum']]= raw.groupby('customer_id')['order_total_num'].agg([
        ('order_total_num_mean', "mean"),
        ('order_total_num_sum', "sum")
    ])    

    # time
    data['order_pay_time_last'] = pd.to_datetime(data['order_pay_time_last'])
    data['order_pay_time_last_m'] = data['order_pay_time_last'].dt.month
    data['order_pay_time_last_d'] = data['order_pay_time_last'].dt.day
    data['order_pay_time_last_h'] = data['order_pay_time_last'].dt.hour
    data['order_pay_time_last_min'] = data['order_pay_time_last'].dt.minute
    data['order_pay_time_last_s'] = data['order_pay_time_last'].dt.second
    data['order_pay_time_last_weekday'] = data['order_pay_time_last'].dt.weekday
    
    # order_pay_time_last diff
    t_min=pd.to_datetime('2012-10-11 00:00:00')
    data['order_pay_time_last_diff'] = (data['order_pay_time_last']-t_min).dt.days
    
    # goods_list_time last diff 
    data['goods_list_time_last'] =pd.to_datetime(data['goods_list_time_last'])    
    data['goods_list_time_diff'] = (data['goods_list_time_last']-t_min).dt.days
    
    # goods_delist_time last diff
    data['goods_delist_time_last'] =pd.to_datetime(data['goods_delist_time_last'])    
    data['goods_delist_time_diff'] = (data['goods_delist_time_last']-t_min).dt.days
    
    # goods_time_diff
    data['goods_time_diff'] =  data['goods_delist_time_diff']-data['goods_list_time_diff']
    return data
 
# preprocess data
train_data = preprocess(train_raw)
train_data['label'] = train_data.index.map(lambda x:int(x in label_raw))
train_data.drop(['goods_list_time_last','goods_delist_time_last','order_pay_time_last'],axis=1,inplace=True)
 
test_data = preprocess(raw)
test_data.drop(['goods_list_time_last','goods_delist_time_last','order_pay_time_last'],axis=1,inplace=True)

print("processed finish")


# save preprocessed data
import pickle
train_data.to_pickle('./train_data.pkl')
test_data.to_pickle('./test_data.pkl')
 
with open('./train_data.pkl', 'rb') as file:
    train_data = pickle.load(file)
 
with open('./test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)
 
train_data = train_data.reset_index()
test_data = test_data.reset_index()
all_df = pd.concat([train_data, test_data], axis=0)
train_data = all_df[all_df['label'].notnull()]
test_data = all_df[all_df['label'].isnull()]

print("start modeling")
import lightgbm as lgb
clf = lgb.LGBMClassifier(
            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2021,
            n_estimators=2500, subsample=1, colsample_bytree=1,
        )
clf.fit(train_data.drop(['label','customer_id'],axis=1),train_data['label'])
 
 
cols=train_data.columns.tolist()
cols.remove('label')
cols.remove('customer_id')
 
y_pred = clf.predict_proba(test_data.drop(['label','customer_id'],axis=1))[:,1] 
result = pd.read_csv('./submission.csv')
result['result'] = y_pred
final_result = result.sort_values('result',ascending=False).copy()
buy_num = 450000
final_result.index=range(len(final_result))
final_result.loc[result.index <= buy_num,'result'] = 1
final_result.loc[result.index > buy_num,'result'] = 0
final_result.sort_values('customer_id',ascending=True,inplace=True)
final_result.to_csv('./no_mutual_info.csv',index=False)