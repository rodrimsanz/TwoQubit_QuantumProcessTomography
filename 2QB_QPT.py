import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

#we define the pauli matrices

I = [[1, 0], [0, 1]]
X = [[0, 1], [1, 0]]
Y = [[0, -1j], [1j, 0]]
Z = [[1, 0], [0, -1]]
sigm = [I, X, Y, Z]

c = np.zeros((16, 4, 4), dtype=np.complex128)

k = 0

for i in range(4):
    for j in range(4):
        c[k] = np.kron(sigm[i], sigm[j])
        k = k+1


C = np.reshape(c,(4,4,4,4))

# C : matrix with elements (Cij) which are  the tensor product between sigma_i and sigma_j , where sigma = (I,,X,Y,Z)

#We define now the 16 imput states:

zero = [1,0]
one = [0,1]
plus = [(1/2)**(1/2),(1/2)**(1/2)]
plus_i = [(1/2)**(1/2), ((1/2)**(1/2))*1j]
phi = [zero, one, plus_i, plus]

states = np.zeros((16, 4), dtype=np.complex128)

k = 0
for i in range(4):
    for j in range(4):
        states[k] = np.kron(phi[i], phi[j])
        k = k+1

#states stores the states with the following order:  zero-zero , zero-one, zero-plus, zero-i , one-zero, one-one...

#We now define the bra's of measurement projection states (we take complex conjugate since they are bra's): 
#order : ZZ, ZX, ZY; XZ, XX, XY; YZ, YX, YY
phiM1 = [zero, plus ,np.conj(plus_i)]
phiM2 = [one, np.dot(Z,plus),np.dot(Z,np.conj(plus_i))]

Mstates1 = np.zeros((9, 4), dtype=np.complex128)
Mstates2 = np.zeros((9, 4), dtype=np.complex128)
Mstates3 = np.zeros((9, 4), dtype=np.complex128)
Mstates4 = np.zeros((9, 4), dtype=np.complex128)

v = 0
for i in range(3):
    for j in range(3):
        Mstates1[v] = np.kron(phiM1[i], phiM1[j])
        v = v+1

w = 0
for i in range(3):
    for j in range(3):
        Mstates2[w] = np.kron(phiM1[i], phiM2[j])
        w = w+1

t = 0
for i in range(3):
    for j in range(3):
        Mstates3[t] = np.kron(phiM2[i], phiM1[j])
        t = t+1

h = 0
for i in range(3):
    for j in range(3):
        Mstates4[h] = np.kron(phiM2[i], phiM2[j])
        h = h+1


h=0
t=0
v=0
w=0

Mstates = np.concatenate((Mstates1,Mstates2,Mstates3,Mstates4))

#Mstates with the aforementines order, and concatenating for 4 ways to measure; (00,01,10,11)

#multiplying sigmas x states and storing the results in a vector:

TwoSigmaPhi = np.zeros((256, 4), dtype=np.complex128)

h=0
for i in range(16):
    for j in range(16):
        TwoSigmaPhi[h] = np.dot(c[i], states[j])
        h = h+1

TwoSigmaPhiM = np.reshape(TwoSigmaPhi,(16,16,4))

#now, we tke the product of the measurement projection states and TwosigmaPhi

element = np.zeros((256*36), dtype=np.complex128)

l = 0
for i in range(36):
    for j in range(256):
        element[l] = np.dot(Mstates[i],TwoSigmaPhi[j])
        l = l+1

print(len(element))

elementM = np.transpose(np.reshape(element, (36,16,16)),(2,0,1))

#now elementM is a 3rank tensor: C_kli

#We take the prosuct between C_kli, C_klj^*.Then, we use de index mapping kl->n ij->m to obtain the A_nm matrix:
# to obtain the A matrix.
element_ni = np.reshape(elementM,(576,16))

A = np.zeros((576*16*16), dtype=np.complex128)

q = 0
for k in range(576):
    for i in range(16):
        for j in range(16):
            A[q] = element_ni[k][i]*np.conj(element_ni[k][j])
            q = q+1
A_M = np.reshape(A,(16*36,16*16))


#Now A_M is a 576x256 matrix with rank =  256. 

#We obtain now its pseudoinverse:
pinv_A_M = np.round(np.linalg.pinv(A_M),4)



#Now we import the output data (the variable is called ZZ_file since it has been tested with the 2-qubit ZZ gate algorithm)
#We take the average of the data and plug them into a vector

ZZ_file = 'Path_to_datafile'
ZZ_data = np.loadtxt(ZZ_file)

ZZ_P_00 = [[] for _ in range(len(ZZ_data))]
ZZ_P_01 = [[] for _ in range(len(ZZ_data))]
ZZ_P_10 = [[] for _ in range(len(ZZ_data))]
ZZ_P_11 = [[] for _ in range(len(ZZ_data))]

for i in range(len(ZZ_data)):
     ZZ_P_00[i] = ZZ_data[i, 2] 

for i in range(len(ZZ_data)):
     ZZ_P_01[i] = ZZ_data[i, 4]

for i in range(len(ZZ_data)):
     ZZ_P_10[i] = ZZ_data[i, 6]

for i in range(len(ZZ_data)):
     ZZ_P_11[i] = ZZ_data[i, 8]

ZZ_P = np.concatenate(( ZZ_P_00, ZZ_P_01, ZZ_P_10, ZZ_P_11))
ZZ_P_kl = np.transpose(np.reshape(ZZ_P, (36,16)))
ZZ_P_ordered = np.ravel(ZZ_P_kl)

#We multiply the data vector by the pseudo inverse of the A matrix in order to obtain the process matrix:

Chi_two = np.round(np.reshape(np.dot(pinv_A_M,ZZ_P_ordered),(16,16)),3)
print(Chi_two)


#3D plot of the process matrix:

x_labels = ['II', '', '', '', '', 'XX', 'XY', '', '', 'YX', 'YY', '', '', '', '', 'ZZ']
y_labels = ['II', '', '', '', '', 'XX', 'XY', '', '', 'YX', 'YY', '', '', '', '', 'ZZ']

x = np.arange(16)
y = np.arange(16)
Xaxis, Yaxis = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fase = np.angle(Chi_two.flatten())
modulo = np.abs(Chi_two.flatten())
fase[modulo == 0] = np.pi  # Si el módulo es 0, establece la fase en 0
vmin = -np.pi  
vmax = np.pi  
cmap = plt.cm.get_cmap('inferno')
norm = plt.Normalize(vmin, vmax)
scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
colors = scalar_map.to_rgba(fase)

bars = ax.bar3d(Xaxis.flatten(), Yaxis.flatten(), np.zeros_like(modulo).flatten(), 1, 1, modulo.flatten(), color=colors)
cbar = fig.colorbar(scalar_map)
cbar.set_ticks([vmin, vmin + np.pi, vmax]) 
cbar.set_ticklabels(['-$\pi$', '0', '$\pi$'])

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_yticks(y)
ax.set_yticklabels(y_labels)

ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_zlabel('Modulus')
cbar.ax.set_ylabel('Phase')

plt.show()