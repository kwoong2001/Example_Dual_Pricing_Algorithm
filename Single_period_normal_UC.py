# Dual Pricing Algorithm in ISO Markets (IEEE Trans. on Power Systems Vol.32, No.4, July 2017)
# Example Code

# Post Unit Commitment Dual problem 에서 z_i^(*) 부분이 불명확
# Unit commitment model의 commitment 결과 없이도 Dual problem에서 해결이 되어야 할 것으로 보이는데,
# Commitment 결과가 있어야지만 결과를 제대로 계산할 수 있음

from __future__ import print_function
from cmath import inf
import numpy as np
np.float_ = np.float64
from docplex.mp.model import Model
from docplex.util.environment import get_environment

# Pre-data
B = [100, 61]
D = [0,0]
D_Max = [100,30]
C = [40,60,80]
P = [0,0,0]
P_min = [0,10,50]
P_max = [40,200,100]
SU = [500,500,1000]

#Set Parameters
i_d_dim = len(D)
i_g_dim = len(P)

## Primal Model
mdl = Model(name='Dual UC Model (Primal)')   # Model - Cplex에 입력할 Model 이름 입력 및 Model 생성
mdl.parameters.mip.tolerances.mipgap = 0.0001;   # 최적화 계산 오차 설정

i_d = [i for i in range(1,i_d_dim+1)]  # 수요의 dimension
i_g = [i for i in range(1,i_g_dim+1)]  # 공급의 dimension

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
mdl.add_constraint(MS == mdl.sum(B[i-1]*d[i] for i in range(1,i_d_dim+1)) - mdl.sum( (C[i-1]*p[i] + SU[i-1]*z[i]) for i in range(1,i_g_dim+1)))

#(1b)
mdl.add_constraint(mdl.sum(d[i] for i in range(1,i_d_dim+1)) - mdl.sum( p[i] for i in range(1,i_g_dim+1)) == 0 )

#(1c)
mdl.add_constraints(p[i] >= P_min[i-1]*z[i] for i in range(1,i_g_dim+1))
mdl.add_constraints(p[i] <= P_max[i-1]*z[i] for i in range(1,i_g_dim+1))

#(1d)
mdl.add_constraints(d[i] <= D_Max[i-1] for i in range(1,i_d_dim+1))
    
mdl.print_information()

s = mdl.solve(log_output = True)

mdl.get_solve_details()

Data = [v.name.split('_') + [s.get_value(v)] for v in mdl.iter_variables()] # 변수 데이터 저장

print(Data)

# Save commitment data 
z_star = list()
for data in Data:
    if data[0] == "Commitment":
        z_star.append(int(data[2]))
        
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
