import numpy as np
import pandas as pd
from math import exp

class Part2:
    def __init__(self):
        self.adult = pd.read_csv('http://raw.githubusercontent.com/ShahabAsoodeh/3IS3/main/adult_with_pii.csv')
        self.q1_D = self.adult[self.adult['Age'] >= 40].shape[0]
        self.A = self.adult[self.adult['Target'] == '>50K'].shape[0] > self.adult[self.adult['Target'] == '<=50K'].shape[0]
        self.q2_D = int(self.A=='True')
        self.ages = self.adult['Age']

    def laplace_mech(self, v, sensitivity, epsilon):
        return v + np.random.laplace(loc=0,scale=sensitivity/epsilon)

    def bernoulli_mech(self,v,epsilon):
        return v ^ self.generate_coin_toss(epsilon)

    def generate_coin_toss(self,epsilon):
        toss = np.random.uniform()
        if toss > float(1/(1+exp(epsilon))):
            return 0
        return 1

    def compute_error(self,queryq,queryM):
        return (abs(queryM-queryq)/queryq)*100

    def run_Q1(self,epsilon,n):
        avg_error = 0
        for _ in range(n):
            M1_D = self.laplace_mech(self.q1_D,1,epsilon)
            avg_error += self.compute_error(self.q1_D,M1_D)
        print(avg_error/n)
    
    def run_Q2(self,epsilon,n):
        avg_error = 0
        for _ in range(n):
            M2_D = self.bernoulli_mech(self.q2_D,epsilon)
            avg_error += int(M2_D != self.q2_D)
        print((avg_error/n)*100)

P2 = Part2()
P2.run_Q1(0.1,1000)
P2.run_Q1(1,1000)
P2.run_Q1(5,1000)
P2.run_Q2(0.1,1000)
P2.run_Q2(1,1000)
P2.run_Q2(5,1000)
