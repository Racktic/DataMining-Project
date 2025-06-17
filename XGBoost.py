import pandas as pd
import pickle
 
# Load the preprocessed data 
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
# use xgboost to train model
import xgboost as xgb
X_train = train_data.drop(['label', 'customer_id'], axis=1)
X_train.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
X_train.fillna(0, inplace=True)
y_train = train_data['label']

dtrain = xgb.DMatrix(X_train, label=y_train, missing=float('nan'))
# hyperparameters for XGBoost
params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'logloss',  
    'max_depth': 6,  
    'learning_rate': 0.1,  
}
num_round = 100  
model = xgb.train(params, dtrain, num_round)

# Predict on test data
X_test = test_data.drop(['label', 'customer_id'], axis=1)
X_test.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
X_test.fillna(0, inplace=True)
dtest = xgb.DMatrix(X_test, missing=float('nan'))
predictions = model.predict(dtest)

# plot the distribution of predictions
import matplotlib.pyplot as plt
plt.hist(predictions, bins=50, edgecolor='black')
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.savefig('predictions_distribution.png')

result = pd.read_csv('./submission.csv')
result['result'] = predictions
final_result = result.sort_values('result', ascending=False).copy()
buy_num = 450000
final_result.index = range(len(final_result))
final_result.loc[result.index <= buy_num,'result'] = 1
final_result.loc[result.index > buy_num,'result'] = 0
final_result.sort_values('customer_id',ascending=True,inplace=True)
final_result.to_csv('./_xgboost_no_mutual_info.csv',index=False)
