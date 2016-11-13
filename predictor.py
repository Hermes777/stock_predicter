import math
import numpy as np
import tushare as ts
import random as rd
import pylab as pl


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
					print(means[k],nums[k])
				print()
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
		print(mu)
		print(sigma)
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

			print(K)
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

			print("current difference",sum(np.abs(np.array(last)-np.array(sigma))))
			change=sum(np.abs(np.array(last)-np.array(sigma)))

			if np.abs(np.abs(change)-np.abs(last_change))<threshold: break
			last_change=change

					
				
		return alpha,mu,sigma

	def GMM_render(self,vec_list,K):
		t=0
		Theta=[[0]*K for _ in range(len(vec_list))]
		for vec in vec_list:
			phi=[0]*K
			for k in range(K):
				phi[k]=self.alpha[k]/np.sqrt(2*math.pi*self.sigma[k])*np.exp(-(vec-self.mu[k])**2/(2*self.sigma[k]))
			sum_phi=sum(phi)
			for k in range(K):
				Theta[t][k]=phi[k]/sum_phi
			t+=1
		return Theta
		
	def Gaussian_Hmm(self,Theta,N,K):
		
		T=len(Theta)

		#initialization
		#N:states T:observation lengths K:observersion
		pi=[1.0/N]*N
		A=[[rd.random() for y in range(N)] for x in range(N)]
		B=[[rd.random() for y in range(K)] for x in range(N)]
		alpha=[[0.0]*N for x in range(T)]
		beta=[[0.0]*N for x in range(T)]
		Gamma=[[0.0]*N for x in range(T)]
		last_Gamma=np.array(Gamma)
		Epsilon=[[[0.0]*N for _ in range(N)] for x in range(T)]


		Iter=0
		while True:
			#Estimation
			sum_alpha=0.0
			for i in range(N):
				for k in range(K):
					alpha[0][i]+=pi[i]*B[i][k]*Theta[0][k]
				sum_alpha+=alpha[0][i]
			for i in range(N):
				alpha[0][i]/=sum_alpha
			for t in range(1,T):
				sum_alpha=0.0
				for i in range(N):
					alpha[t][i]=0.0
					for j in range(N):
						alpha[t][i]+=alpha[t-1][j]*A[j][i]
					multi=0.0
					for k in range(K):
						multi+=(B[i][k]*Theta[t][k])
					alpha[t][i]*=multi
					sum_alpha+=alpha[t][i]
				for i in range(N):
					alpha[t][i]/=sum_alpha
			
			
			for i in range(N):
				beta[T-1][i]=1.0/N
			for t in range(T-2,-1,-1):
				sum_beta=0.0
				for i in range(N):
					beta[t][i]=0.0
					for j in range(N):
						for k in range(K):
							beta[t][i]+=beta[t+1][j]*A[i][j]*B[j][k]*Theta[t+1][k]
					sum_beta+=beta[t][i]
				for i in range(N):
					beta[t][i]/=sum_beta


			#print alpha
			#print B
			#print alpha
			for t in range(T):
				sum_Gamma=sum(np.array(alpha[t])*beta[t])
				for i in range(N):
					Gamma[t][i]=float(alpha[t][i])*beta[t][i]/sum_Gamma
			for t in range(T-1):
				sum_Epsilon=0.0
				for i in range(N):
					for j in range(N):
						for k in range(K):
							sum_Epsilon+=alpha[t][i]*A[i][j]*beta[t][j]*B[j][k]*Theta[t+1][k]
				for i in range(N):
					for j in range(N):
						Epsilon[t][i][j]=0.0
						for k in range(K):
							Epsilon[t][i][j]+=(float(alpha[t][i])*A[i][j]*beta[t][j]*B[j][k]*Theta[t+1][k])/sum_Epsilon

			sum_Epsilon=[[0.0]*N for _ in range(N)]
			sum_Gamma=[0.0]*N
			for i in range(N):
				for j in range(N):
					for t in range(T):
						sum_Epsilon[i][j]+=Epsilon[t][i][j]
				for t in range(T):
					sum_Gamma[i]+=Gamma[t][i]

			#print(Gamma)
			print()
			#print(sum_Epsilon)
			print()
			print(np.sum(np.abs(Gamma-last_Gamma)/np.sum(np.abs(Gamma))))

			for i in range(N):
				pi[i]=Gamma[0][i]
			for i in range(N):
				for j in range(N):
					A[i][j]=sum_Epsilon[i][j]/sum_Gamma[i]

			for i in range(N):
				for j in range(K):
					B[i][j]=0.0
					for t in range(T):
						B[i][j]+=Gamma[t][i]*Theta[t][j]
					B[i][j]/=sum_Gamma[i]
			
			if np.sum(np.abs(Gamma-last_Gamma)/np.sum(np.abs(Gamma)))<0.00001:
				break
			#Maximizarion
			last_Gamma=np.array(Gamma)
			Iter+=1


		cluster=[0]*T
		num=[0]*N
		tot=[0.0]*N
		for t in range(T):
			cluster[t]=np.argmax(Gamma[t])
		print(cluster)
		for t in range(T):
			if(cluster[t]==0):
				pl.plot(t,self.price[t],'or')
			if(cluster[t]==1):
				pl.plot(t,self.price[t],'ob')
			if(cluster[t]==2):
				pl.plot(t,self.price[t],'oy')
			if(cluster[t]==3):
				pl.plot(t,self.price[t],'og')
			if(cluster[t]==4):
				pl.plot(t,self.price[t],'ob')
			tot[cluster[t]]+=self.price_delta[t]
			num[cluster[t]]+=1
		pl.show()
		print('T',T)
		for i in range(N):
			if num[i]!=0:
				print("state",i," average:",tot[i]/num[i])
		return A,B,pi,cluster



	def train(self):
		data=ts.get_h_data('002496',start='2014-06-01',autype='qfq')
		
		st=50
		self.price=data['close'][:2000+st]*100
		self.volume=data['volume'][st:1000+st]
		self.price_delta=np.array(self.price[:-1])-np.array(self.price[1:])
		self.price_delta=list(self.price_delta)
		self.price_delta.reverse()
		print self.price
		self.price=list(self.price)
		self.price.reverse()
		self.volume_delta=np.log(self.volume[1:])-np.log(self.volume[:-1])

		_=0
		for x in range(100):
			print(self.price[x],data['close'][st:1000+st][x],self.price_delta[x],_)
			_+=1
		self.alpha,self.mu,self.sigma=self.GMM(self.price_delta[:2000],8)
		#print self.alpha,self.mu,self.sigma
		observation=self.GMM_render(self.price_delta[:2000],8)

		#observation=[[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1],[0.1,0.9],[0.9,0.1]]
		print(len(data['open']))
		self.A,self.B,self.pi,cluster=self.Gaussian_Hmm(observation,3,8)


		print(self.A)
		print(self.B)
		print(self.pi)


	def Gaussian_Hmm_viterbi(self,Theta,N,K):
		T=len(Theta)
		alpha=[[0.0]*N for x in range(T)]
		sum_alpha=0.00
		for i in range(N):
			for k in range(K):
				alpha[0][i]+=self.pi[i]*self.B[i][k]*Theta[0][k]
			sum_alpha+=alpha[0][i]
		for i in range(N):
			alpha[0][i]/=sum_alpha
		for t in range(1,T):
			sum_alpha=0.0
			for i in range(N):
				alpha[t][i]=0.0
				for j in range(N):
					if alpha[t-1][j]*self.A[j][i]>alpha[t][i]:
						alpha[t][i]=alpha[t-1][j]*self.A[j][i]
				multi=0.0
				for k in range(K):
					multi+=(self.B[i][k]*Theta[t][k])
				alpha[t][i]*=multi
				sum_alpha+=alpha[t][i]
			for i in range(N):
				alpha[t][i]/=sum_alpha
			ans=np.argmax(alpha[t])
			print ans,t
		


	def test(self):
		st=50
		data=ts.get_hist_data('002496')
		test_price=data['close'][:st]*100
		test_price_delta=data['price_change'][:st]*100
		test_price_delta=list(test_price_delta)
		test_price_delta.reverse()
		print test_price
		test_price=list(test_price)
		test_price.reverse()

		observation=self.GMM_render(test_price_delta,8)
		self.Gaussian_Hmm_viterbi(observation,3,8)
		


		
		o=0
		
def main():
	e=predictor("000002","hmm")
	e.train()
	e.test()

if __name__ == '__main__':
	main()