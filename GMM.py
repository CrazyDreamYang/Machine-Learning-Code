#模拟两个正态分布的参数
 
from numpy import *
import numpy as np
import random
import copy
 
SIGMA = 6
EPS = 0.0001
#均值不同的样本
#生成样本，权重可以通过控制数据个数来控制
def generate_data():	
    n_samples = 10000
    X = np.zeros((n_samples,1))
    X = np.concatenate([np.random.normal(-2, 1, int(0.3 * n_samples)),np.random.normal(3, 1, int(0.7 * n_samples))]).reshape(-1, 1)
    return X
 
#EM算法
def my_GMM(X):
    k = 2
    N = len(X)
    # 根据数据初始化参数
    Miu = np.random.rand(k,1)
    Miu = np.random.choice(X.flatten(), k)
    Posterior = mat(zeros((N,k)))	
    sigma = np.random.rand(k,1)
    sigma = np.random.random_sample(size=k) * X.var()
    alpha = np.random.rand(k,1)
    alpha = np.ones(k) / k
    dominator = 0
    numerator = 0
    #先求后验概率

    for it in range(1000):
        #E-step
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2)
            for j in range(k):
                numerator = np.exp(-1.0/(2.0*sigma[j]) * (X[i] - Miu[j])**2)
                Posterior[i,j] = numerator/dominator			
        oldMiu = copy.deepcopy(Miu)
        oldalpha = copy.deepcopy(alpha)
        oldsigma = copy.deepcopy(sigma)
        #M-step
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i,j] * X[i]
                dominator = dominator + Posterior[i,j]
            Miu[j] = numerator/dominator
            alpha[j] = dominator/N
            tmp = 0
            for i in range(N):
                tmp = tmp + Posterior[i,j] * (X[i] - Miu[j])**2
            sigma[j] = tmp/dominator
            
        if ((abs(Miu - oldMiu)).sum() < EPS) and \
            ((abs(alpha - oldalpha)).sum() < EPS) and \
            ((abs(sigma - oldsigma)).sum() < EPS):
                break
    print(alpha)
    print(sigma)
    print(Miu)

if __name__ == '__main__':
    X = generate_data()
    my_GMM(X)
