import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

def plot(df,x,y,overlay = False):
	if overlay:
		sns.regplot(x=x, y=y,data = df , ax=ax)
	else :
		sns.regplot(x=x, y=y, data = df)
		plt.show()

def zscore(col):
	mean = col.mean()
	std = col.std()
	return (col-mean)/std


if __name__ == '__main__':

	file = sys.argv[1]
	df = pd.read_csv(file, sep = "\t", header = None , names = ["Packet Size", "RTT"])
	df.dropna(inplace = True)
	columns = df.columns



	plot(df,"Packet Size", "RTT")

	df["Min RTT"] = df.groupby('Packet Size').RTT.transform(np.min)
	
	filtered = np.unique(df[["Packet Size", "Min RTT"]],axis = 0)
	new_df = pd.DataFrame({"Packet Size" : filtered[:,0],"Min RTT" : filtered[:,1]})

	plot(new_df,"Packet Size", "Min RTT")
	new_df['zscore'] = zscore(new_df["Min RTT"])
	new_df = new_df[new_df["zscore"] < 1]
	plot(new_df, "Packet Size", "Min RTT")


	fig, ax = plt.subplots()
	plot(new_df, "Packet Size", "Min RTT", overlay = True )

	clf = LinearRegression()
	clf.fit(new_df["Packet Size"].values.reshape(-1, 1),new_df["Min RTT"].values.reshape(-1, 1))
	predict = np.linspace(0,35000, 10).reshape(-1,1)
	prediction = clf.predict(predict)
	pred_df = pd.DataFrame({"Packet Size": predict.flatten(), "Min RTT" : prediction.flatten()})
	plot(pred_df,"Packet Size", "Min RTT", overlay = True )

	plt.legend(labels=["Raw Data", f"LR {clf.coef_.flatten()[0]}x + {clf.intercept_[0]}"])
	plt.show()