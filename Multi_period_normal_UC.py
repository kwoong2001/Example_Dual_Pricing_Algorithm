# Dual Pricing Algorithm in ISO Markets (IEEE Trans. on Power Systems Vol.32, No.4, July 2017)
# Modified Example Code for multi-period

# 조건 설계 필요

from __future__ import print_function
from cmath import inf
import numpy as np
np.float_ = np.float64
from docplex.mp.model import Model
from docplex.util.environment import get_environment

# Pre-data
B = [[100, 61, 50],[100, 80, 80]]
D = [[0,0,0],[0,0,0]]
D_Max = [100,30,50]
C = [[40,60,70],[40,60,70]]
P = [[0,0,0], [0,0,0]]
P_min = [0,10,20]
P_max = [40,100,50]
SU = [500,500,500]

#Set Parameters
i_d_dim = len(D_Max)
i_g_dim = len(P_max)

## Primal Model
mdl = Model(name='Dual UC Model (Primal)')   # Model - Cplex에 입력할 Model 이름 입력 및 Model 생성
mdl.parameters.mip.tolerances.mipgap = 0.0001;   # 최적화 계산 오차 설정

time = [t for t in range(1,len(B)+1)]
i_d = [(t,i) for t in range(1,len(B)+1) for i in range(1,i_d_dim+1)]  # 수요의 dimension
i_g = [(t,i) for t in range(1,len(P)+1) for i in range(1,i_g_dim+1)]  # 공급의 dimension

# Variables (Cont. and Bin.)
d = mdl.continuous_var_dict(i_d,lb = 0, ub = inf, name = "Demand")
p = mdl.continuous_var_dict(i_g,lb = 0, ub = inf, name = "Supply")
z = mdl.binary_var_dict(i_g, name = "Commitment")

#Functions
MS = mdl.continuous_var(lb = -inf,ub =inf,name = "Social Welfare") # Primal Problem

# Objective funcion (1a)
mdl.maximize(MS)

# Constraint 

#(1a)
mdl.add_constraint(MS == mdl.sum(B[t-1][i-1]*d[(t,i)] for t in range(1, len(time)+1) for i in range(1,i_d_dim+1)) - mdl.sum( (C[t-1][i-1]*p[(t,i)] + SU[i-1]*z[(t,i)]) for t in range(1, len(time)+1) for i in range(1,i_g_dim+1)))


#(1b)
mdl.add_constraints((mdl.sum(d[(t,i)] for i in range(1,i_d_dim+1)) - mdl.sum( p[(t,i)] for i in range(1,i_g_dim+1)) == 0 for t in range(1, len(time)+1)),'1b')

#(1c)
mdl.add_constraints((p[(t,i)] >= P_min[i-1]*z[(t,i)] for t in range(1, len(time)+1) for i in range(1,i_g_dim+1)),'1c_min')
mdl.add_constraints((p[(t,i)] <= P_max[i-1]*z[(t,i)] for t in range(1, len(time)+1) for i in range(1,i_g_dim+1)),'1c_max')

#(1d)
mdl.add_constraints((d[(t,i)] <= D_Max[i-1] for t in range(1, len(time)+1) for i in range(1,i_d_dim+1)),'1d')
    
mdl.print_information()

s = mdl.solve(log_output = True)

mdl.get_solve_details()

Data = [v.name.split('_') + [s.get_value(v)] for v in mdl.iter_variables()] # 변수 데이터 저장

print(Data)

# Save commitment data 
z_star = P.copy()
for data in Data:
    if data[0] == "Commitment":
        z_star[int(data[1])-1][int(data[2])-1] = int(data[3]) 
        
## Dual Model
mdl = Model(name='Dual UC Model (Dual)')   # Model - Cplex에 입력할 Model 이름 입력 및 Model 생성
mdl.parameters.mip.tolerances.mipgap = 0.0001;   # 최적화 계산 오차 설정

# Variables (Cont. and Bin.)
lamb = mdl.continuous_var(lb = 0, ub = inf, name = "Lambda")
alpha_max = mdl.continuous_var_dict(i_d,lb = 0, ub = inf, name = "Alpha_max")
delta = mdl.continuous_var_dict(i_g,lb = -inf, ub = inf, name = "Delta")
beta_min = mdl.continuous_var_dict(i_g,lb = 0, ub = inf, name = "Beta_min")
beta_max = mdl.continuous_var_dict(i_g,lb = 0, ub = inf, name = "Beta_max")

#Functions
RC = mdl.continuous_var(lb =0,ub =inf,name = "Total commitment cost") # Dual Problem

# Objective funcion (2a)
mdl.minimize(RC)

# Contraint (2a)
mdl.add_constraint(RC == mdl.sum( D_Max[i-1]*alpha_max[i] for i in range(1,i_d_dim+1)) + mdl.sum( z_star[i-1] * delta[i] for i in range(1,i_g_dim+1)) )

#constraint (2b)
mdl.add_constraints(lamb + alpha_max[i] >= B[i-1] for i in range(1,i_d_dim+1))

#constraint (2c)
mdl.add_constraints((-1)*lamb + beta_max[i] - beta_min[i] >= (-1)*C[i-1] for i in range(1,i_g_dim+1))

#constraint (2d)
mdl.add_constraints(delta[i] - P_max[i-1]*beta_max[i] + P_min[i-1]*beta_min[i] == (-1)*SU[i-1] for i in range(1,i_g_dim+1))

mdl.print_information()

s = mdl.solve(log_output = True)

mdl.get_solve_details()

Data = [v.name.split('_') + [s.get_value(v)] for v in mdl.iter_variables()] # 변수 데이터 저장

print(Data)
