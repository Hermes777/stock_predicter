from flask import Flask
import tushare as ts
app = Flask(__name__)

@app.route('/')
def hello_world():
	s="	<body>\n"
	s+="<p>This is my first paragraph.\n</p> </body>"
	return s

@app.route('/stock/<post_id>')
def show_post(post_id):
	# show the post with the given id, the id is an integer
	_index=ts.get_hist_data(post_id)
	temp="<body>\n"
	temp+="<p>open:"+str(_index['open'][0])+"</p>"
	temp+="<p>close:"+str(_index['close'][0])+"</p>"
	temp+="<p>ma5:"+str(_index['ma5'][0])+"</p>"
	temp+="<p>ma10:"+str(_index['ma10'][0])+"</p>"
	temp+="<p>ma20:"+str(_index['ma20'][0])+"</p>"
	temp+="</body>"
	return render_template('dsv.html', post_id=post_id)

if __name__ == '__main__':
	app.run()