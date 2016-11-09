import math
import numpy as np
import tushare as ts


class predictor:


	def __init__(self,path,method):
		self.path=path
		self.method=method
		self.price=[]
		self.volume=[]
		self.price_delta=[]
		self.volume_delta=[]

	def kmeans(self,vec,K):
		mu=[0]*K
		N=len(vec)
		last=[sum(vec)/N]
		K_log=int(np.log2(K))
		cores=[0]*2*K
		K=1
		for num_cores in range(K_log):
			cores=[0]*2*K
			for x in range(K):
				cores[x]=last[x]-0.001
				cores[x+K]=last[x]+0.001
			K*=2
			last=[0]*K
			while True:
				means=[0]*K
				nums=[0]*K
				for j in range(N):
					k=np.argmin(np.abs(np.array(cores)-vec[j]))
					means[k]+=vec[j];
					nums[k]+=1;
				for k in range(K-1):
					print means[k],nums[k]
				print ""
				for k in range(K):
					cores[k]=means[k]/nums[k]

				if(sum(np.abs(np.array(last)-np.array(cores)))<0.01):
					break

				for k in range(K):
					last[k]=cores[k]
		
		means=[0]*K
		sigma=[0]*K
		nums=[0]*K
		Gamma=[[0]*K for x in range(N)]
		
		for j in range(N):
			k=np.argmin(np.abs(np.array(cores)-vec[j]))
			means[k]+=vec[j];
			sigma[k]+=(vec[j]-cores[k])**2;
			Gamma[j][k]=1.0

			nums[k]+=1;
		#the method processing sigma is questionable.
		for k in range(K):
			means[k]=means[k]/nums[k]
			sigma[k]=sigma[k]/nums[k]
			
		return means,sigma,Gamma




	def GMM(self,vec,K):
		N=len(vec)
		phi=[[0]*K for x in range(N)]
		alpha=[0]*K
		mu=[0]*K
		#preprocessing

		mu,sigma,Gamma=self.kmeans(vec,K)	
		print mu
		print sigma
		alpha=list(np.array([sum([Gamma[j][k] for j in range(N)]) for k in range(K)])/N)

		#iteration begins
		threshold=0.1
		last_change=0
		while True:
			#Estimationsigma
			for j in range(N):
				for k in range(K):
					phi[j][k]=alpha[k]/np.sqrt(2*math.pi*sigma[k])*np.exp(-(vec[j]-mu[k])**2/(2*sigma[k]))
			#print phi
			for j in range(N):
				sum_phi=sum(phi[j])
				for k in range(K):
					Gamma[j][k]=phi[j][k]/sum_phi;

			#Maximization
			last=[x for x in sigma]

			print K
			alpha=list(np.array([sum([Gamma[j][k] for j in range(N)]) for k in range(K)])/N)
			mu=[0]*K
			for k in range(K):
				for j in range(N):
					mu[k]+=Gamma[j][k]*vec[j]
				mu[k]=float(mu[k])/alpha[k]/N;

			sigma=[0]*K
			for k in range(K):
				for j in range(N):
					sigma[k]+=Gamma[j][k]*(vec[j]-mu[k])**2
				sigma[k]=float(sigma[k])/alpha[k]/N;

			print "current difference",sum(np.abs(np.array(last)-np.array(sigma)))
			change=sum(np.abs(np.array(last)-np.array(sigma)))

			if np.abs(np.abs(change)-np.abs(last_change))<threshold: break
			last_change=change

					
				
		return alpha,mu,sigma

	def Count_Gaussian_pos(vec,K):
		phi=[0]*K
		for k in range(K):
			phi[k]=alpha[k]/np.sqrt(2*math.pi*sigma[k])*np.exp(-(vec-mu[k])**2/(2*sigma[k]))
		sum_phi=sum(phi)
		for k in range(K):
			Gamma[k]=phi[k]/sum_phi
		return Gamma
		
	def Gaussian_Hmm(O,alp,N,K):
		
		T=len(O)
		#initialization
		#N:states T:observation lengths K:observersion
		pi=[1.0/N]*N
		A=[[1.0/N]*N for x in range(N)]
		B=[[1.0/K]*K for x in range(N)]


		while True:
			#Estimation
			for i in range(N):
				alpha[0][i]=pi[i]
			for t in range(1,T):
				for i in range(N):
					alpha[t][i]=0
					for j in range(N):
						alpha[t][i]+=alpha[t-1][j]*A[j][i]*B[i][O[t]]
			
			for i in range(N):
				beta[T-1][i]=1
			for t in range(T-2,-1,-1):
				for i in range(N):
					beta[t][i]=0
					for j in range(N):
						beta[t][i]+=beta[t+1][j]*A[i][j]*B[j][O[t+1]]


			for t in range(T):
				sum_Gamma=sum(np.array(alpha[t]))
				for i in range(N):
					Gamma[t][i]=alpha[t][i]*beta[t][i]/sum_Gamma
			for t in range(T-1):
				sum_Epsilon=0.0
				for i in range(N):
					for j in range(N):
						sum_Epsilon+=alpha[t][i]*A[i][j]*beta[t][j]*B[j][O[t+1]]
				for i in range(N):
					for j in range(N):
						Epsilon[t][i][j]=alpha[t][i]*A[i][j]*beta[t][j]*B[j][O[t+1]]/sum_Epsilon

			#Maximizarion
			sum_Epsilon=[[0]*N for _ in range(N)]
			sum_Gamma=[0]*N
			for i in range(N):
				for j in range(N):
					for t in range(T):
						sum_Epsilon[i][j]+=Epsilon[t][i][j]
				for t in range(T):
					sum_Gamma[i]+=Gamma[t][i]

			for i in range(N):
				pi[i]=Gamma[0][i]
			for i in range(N):
				for j in range(N):
					A[i][j]=sum_Epsilon[i][j]/sum_Gamma[i]

			for i in range(N):
				for j in range(K):
					B[i][j]=0.0
					for t in range(T):
						if O[t]==j:
							B[i][j]+=Gamma[i][t]
					B[i][j]/sum_Gamma[i]
		return A,B,pi



	def train(self):
		data=ts.get_hist_data(self.path)
		
		st=50
		self.price=data['open'][st:1000+st]
		self.volume=data['price_change'][st:1000+st]
		self.price_delta=data['price_change'][st:999+st]*100
		self.price_delta.reverse()
		self.volume_delta=np.log(self.volume[1:])-np.log(self.volume[:-1])

		_=0
		for x in range(100):
			print self.price[x],data['close'][st:1000+st][x],self.price_delta[x],_
			_+=1
		self.alpha,self.mu,self.sigma=self.GMM(self.price_delta,8)
		print self.alpha,self.mu,self.sigma

		self.A,self.B,self.pi=Gaussian_Hmm(self.price_delta,3,8)




	def test(self):
		data=ts.get_hist_data(self.path)
		st=50
		self.price=data['open'][st:1000+st]
		self.volume=data['price_change'][st:1000+st]
		self.price_delta=data['price_change'][st:999+st]*100
		
		o=0
		
def main():
	e=predictor("002496","hmm")
	e.train()

if __name__ == '__main__':
	main()