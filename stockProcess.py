import tushare as ts

def encoder(_index, _date,_long):
	temp=[]
	order=[]
	temp.append(_index['open'][_date])
	temp.append(_index['close'][_date])
	temp.append(_index['ma5'][_date])
	temp.append(_index['ma10'][_date])
	temp.append(_index['ma20'][_date])
	order=temp.copy()
	order.sort()
	cluster=[0]*5
	cluster[temp.index(order[0])]=0
	now=0
	for i in range(1,5):
		if(abs(order[i]-order[i-1]/order[i])<0.005):
			cluster[temp.index(order[i])]=now;
		else:
			now+=1
			cluster[temp.index(order[i])]=now;
	cnt=0
	for i in range(0,5):
		cnt=cnt*5+cluster[i];
	if(abs(_index['open'][_date]-_long)/_long<0.01):
		cnt=cnt*3+1
	elif(_index['open'][_date]>_long):
		cnt=cnt*3
	elif(_index['open'][_date]<_long):
		cnt=cnt*3+2
	return cnt


def main():
	_index=ts.get_hist_data('sh')
	_long_index=ts.get_hist_data('sh')
	#_date=5
	for _date in range(0,20):
		t=encoder(_index, _date,_long_index['ma20'][_date])
		print(t)

if __name__ == '__main__':
	main()