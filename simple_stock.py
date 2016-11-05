import tushare as ts
import sys  
  
for arg in sys.argv:
    a=arg
#c=ts.get_hist_data()
b=ts.get_today_all()
change={}
i=0

for value in list(b['code']):
    change[value]=b['changepercent'][list(b['code']).index(value)],b['turnoverratio'][list(b['code']).index(value)],b['per'][list(b['code']).index(value)]
    i+=1
change= sorted(change.items(), key=lambda d:d[1][1], reverse = False)
change= change[-300:]
print(change)

for (d,x) in change:
	if x[1]>1 and x[0]<0 and x[2]>0 and x[2]<13:
		print(d)
print(b.keys())
print(b['changepercent'][list(b['code']).index(a)],b['trade'][list(b['code']).index(a)],b['pb'][list(b['code']).index(a)],b['per'][list(b['code']).index(a)],b['mktcap'][list(b['code']).index(a)],b['nmc'][list(b['code']).index(a)])

