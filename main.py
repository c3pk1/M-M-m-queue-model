import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import heapq

class K_FCFS_simulator:

  def __init__(self, sample_size, lambda_, mu_, K = 1, seed=0):

    self.sample_size = sample_size
    random.seed(seed)
    self.arrival_time_generator = lambda: -math.log(np.random.random())/lambda_
    self.service_time_generator = lambda: -math.log(np.random.random())/(mu_/K) 
    self.K = K

  def simulate(self):

    t = 0                                   # 現在時刻
    SS = 0                                  # 系内客数
    N_A = 0                                 # 到着客数
    N_D = 0                                 # 離脱客数
    t_A = self.arrival_time_generator()     # 次の到着時刻
    t_Dl = [(math.inf, -1)] * self.K        # 次の離脱時刻 (時刻と到着した客のidを記録)

    A = [0] * (self.sample_size*2+1)        # i番目の客の到着時刻を記録するための配列
    D = [math.inf] * (self.sample_size*2+1) # i番目の客の離脱時刻を記録するための配列
    
    while N_D != self.sample_size:
      top = heapq.heappop(t_Dl)
      t_D = top[0]
      customer_id = top[1]
      if t_A <= t_D:
        heapq.heappush(t_Dl, (t_D, customer_id))
        t = t_A
        N_A = N_A + 1
        if N_A <= self.sample_size*2: 
          A[N_A] = t
        
        SS = SS + 1
        X = self.arrival_time_generator()
        t_A = t + X                           # 次の客の到着時間を設定

        if SS <= self.K:                      # 系内客数がK以下(サーバに空きがある時)
          Y = self.service_time_generator()   
          t_D = t + Y                         # N_Aに対して, 新しくサービス提供時間を設定する
          heapq.heappush(t_Dl, (t_D, N_A))
      else:
        t = t_D
        N_D = N_D + 1
        D[customer_id] = t
        SS = SS - 1

        if SS >= self.K:                     # 系内客数がKより大きい場合, 
          Y = self.service_time_generator()  # queueの先頭(N_D+self.K)にサービス時間を設定する
          t_D = t + Y
          heapq.heappush(t_Dl, (t_D, N_D + self.K))           

    # シミュレーション結果から平均滞在時間を求める
    mean_sojourn_time = 0
    for i in range(1,self.sample_size*2+1):
      if D[i] == math.inf:
        continue
      mean_sojourn_time += D[i] - A[i]
    mean_sojourn_time /= self.sample_size

    return mean_sojourn_time

def simulate(server_size, sample_size):

  # シミュレーションに関する定数の設定
  customer_size = 100000       # 1シミュレーションで扱う客数の数

  x1 = []
  y1 = []
  confidence_interval = []
  
  for rho in np.arange(0.1, 1.0, 0.1):
    average_arrival_rate = rho # 平均到着率(λ)
    average_service_rate = 1.0 # 平均サービス率(μ)

    # sample_sizeの数だけシミュレーションを行う
    result = [] # 各シミュレーションでの平均滞在時間を管理する
    for _ in range(sample_size):
      model = K_FCFS_simulator(sample_size=customer_size, lambda_=average_arrival_rate, mu_=average_service_rate, K=server_size, seed=random.randint(1, 100000000))
      mean_sojourn_time = model.simulate()
      result.append(mean_sojourn_time)

    # 信頼区間の計算
    result = np.array(result, dtype=np.float64)
    sample_mean = np.mean(result, dtype=np.float64)
    sample_std = np.std(result, ddof=1, dtype=np.float64) # 自由度n-1で割る

    print(sample_mean)
    if sample_size == 11:
      t = 2.228
    elif sample_size == 21:
      t = 2.086
    else:
      t = 2.009
    
    x1.append(rho)
    y1.append(sample_mean)
    confidence_interval.append(t * sample_std/math.sqrt(sample_size))
    
  # シミュレーション結果の描画
  g = plt.errorbar(x1, y1, yerr=confidence_interval, elinewidth=0.8, ecolor='black', markersize=5, fmt='xk', capsize=2)
  g[-1][0].set_linestyle(':') # 信頼区間のlinestyleの設定

  # 平均滞在時間の理論値(リトルの公式から)の描画 
  x2 = np.arange(0.0, 1.0, 0.01)
  y2 = []
  for rho in x2:
    p0 = np.sum([((rho * server_size) ** k) / math.factorial(k) for k in range(server_size)]) + ((rho * server_size) ** server_size) / math.factorial(server_size) / (1-rho)
    p0 = 1/p0
    p0 *= ((rho * server_size) ** server_size) / math.factorial(server_size)  / ((1 - rho) ** 2)
    p0 += server_size
    y2.append(p0)

  plt.plot(x2, y2, linestyle='-', linewidth=0.8, color='black')
  
  plt.xlim(0, 1)
  plt.title('mean sojourn time(M/M/m) (m = {})'.format(server_size))
  plt.xlabel(r'$\rho=\lambda/m\mu$')
  plt.ylabel(r'$\rm{E}[\rm{T}]$')
  
  plt.savefig('sim.png')
  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Simulate M/M/m queueing model. Output a graph of mean sojourn time obtained from simulation and little\'s law.')
  parser.add_argument(
    'm',
    type=int,
    help='the number of server',
    default = 1
  )
  parser.add_argument(
    '-s',
    type=int,
    help='the number of sample_size',
    default = 51,
    choices = [11, 21, 51]
  )

  args = parser.parse_args()
  m = args.m
  sample_size = args.s
  simulate(m, sample_size)
