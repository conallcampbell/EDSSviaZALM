#Building 3 protocols of entanglement distribution:
    #Direct ED
    #Cubitt
    #Fedrizzi
#and testing their performance in the presence of depolarising noise

import numpy
import scipy
import math
import cmath
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


#kronecker product code for more than 2 inputs
def kron(*matrices):
    result = numpy.array([[1]])
    for matrix in matrices:
        result = numpy.kron(result, matrix)
    return result

#Partial Trace Code
def IntegerDigits(n, b, l):
    digits = [0] * l
    pos = l - 1
    while pos != -1:
        digits[pos] = int(n % b)
        n //= b
        pos -= 1
    return digits

def FromDigits(digits, base):
    digits = digits[::-1]
    n = 0
    for i, d in enumerate(digits):
        n += d * base**i
        
    return n

def SwapParts(digits, p1, p2):
    new = numpy.copy(digits)
    new[p1] = digits[p2]
    new[p2] = digits[p1]
    return new

def dTraceSystem(D,s,dimen):
    Qudits=sorted(s)
    Qudits.reverse()
    TrkM = D
    z=len(Qudits)
    
    for q in range(z):
        n=math.log(TrkM.shape[0],dimen)
        assert n % 1 == 0
        n = int(n)
        
        M=TrkM
        k=Qudits[q]
        temp = numpy.zeros(M.shape[0], dtype=complex)
        if k!=n:
            for j in range(n-k):
                b={0}
                for i in range(dimen**n):
                    digits=IntegerDigits(i,dimen,n)
                    if digits[n-1] != digits[n-j-2] and i not in  b:
                        number=FromDigits(
                            SwapParts(digits, n-1, n-j-2),
                            dimen
                        )
                        b.add(number)

                        temp[:] = M[i, :]
                        M[i, :] = M[number, :]
                        M[number, :] = temp

                        temp[:] = M[:, i]
                        M[:, i] = M[:, number]
                        M[:, number] = temp
        
        TrkM=[]
        for p in range(0,dimen**n,dimen):
            TrkM.append(
                sum(
                    M[p+h, h:dimen**n:dimen]
                    for h in range(dimen)
                )
            )
        TrkM = numpy.array(TrkM)
    
    return TrkM
#Recall matrix as dTraceSystem(matrix,[systems I want to trace out],dimension of system)





#defining basis vectors

#zero vector and its conjugate transpose                                         
zero=numpy.array([[1],[0]])
zeroCT=numpy.conjugate(zero.T)
#one vector and its conjugate transpose
one=numpy.array([[0],[1]])
oneCT=numpy.conjugate(one.T)
#plus vector and its conjugate transpose
plus=numpy.array([[1],[1]])*1/math.sqrt(2)
plusCT=numpy.conjugate(plus.T)
#minus vector and its conjugate transpose
minus=numpy.array([[1],[-1]])*1/math.sqrt(2)
minusCT=numpy.conjugate(minus.T)
#plusy and its conjugate transpose
plusy=numpy.array([[1],[complex(0.0, 1)]])*1/math.sqrt(2)
plusyCT=numpy.conjugate(plusy.T)
#minusy and its conjugate transpose
minusy=numpy.array([[1],[complex(0.0, -1)]])*1/math.sqrt(2)
minusyCT=numpy.conjugate(minusy.T)
#defining the Bell states
phiplus=(numpy.kron(zero, zero)+numpy.kron(one, one))*1/math.sqrt(2)
phiminus=(numpy.kron(zero, zero)-numpy.kron(one, one))*1/math.sqrt(2)
psiplus=(numpy.kron(zero, one)+numpy.kron(one, zero))*1/math.sqrt(2)
psiminus=(numpy.kron(zero, one)-numpy.kron(one, zero))*1/math.sqrt(2)
#defining the outer product of the Bell states
Phiplus=phiplus@numpy.conjugate(phiplus.T)
Phiminus=phiminus@numpy.conjugate(phiminus.T)
Psiplus=psiplus@numpy.conjugate(psiplus.T)
Psiminus=psiminus@numpy.conjugate(psiminus.T)
#defining Pauli matrices
pauliX=numpy.array([[0,1],[1,0]])
pauliY=numpy.array([[0,complex(0,-1)],[complex(0,1),0]])
pauliZ=numpy.array([[1,0],[0,-1]])
#Defining the Kraus operators for the depolarising channel
def depolarising_kraus0(p):
    return math.sqrt(1-p)*numpy.identity(2)
def depolarising_kraus1(p):
    return math.sqrt(p/3)*pauliX
def depolarising_kraus2(p):
    return math.sqrt(p/3)*pauliY
def depolarising_kraus3(p):
    return math.sqrt(p/3)*pauliZ




# To save computation time this file will save only all the output data
# a separate file will be created to use this output data and obtain a plot
# We use the following function to accomplish this
def write_out(x , y , filename):
    file = open(filename , 'w')
    for ii in range (0 , len(x)):
        string_to_be_written = str(x[ii]) + " " + str(y[ii]) + "\n"
        file.write(string_to_be_written)
    return    




#Direct Entanglement Distribution



#in this protocol, a maximally entangled source is created and sent to the sites of Alice and Bob
#in our protocol we consider the Phiplus
directEDstate=Phiplus

#Applying the depolarising channel
def depolarised_directED(p):
    return (
            (
                (
                    kron(depolarising_kraus0(p) , depolarising_kraus0(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus0(p) , depolarising_kraus0(p))).T)
                ) +
                (
                    kron(depolarising_kraus0(p) , depolarising_kraus1(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus0(p) , depolarising_kraus1(p))).T)
                ) +
                (
                    kron(depolarising_kraus0(p) , depolarising_kraus2(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus0(p) , depolarising_kraus2(p))).T)
                ) +
                (
                    kron(depolarising_kraus0(p) , depolarising_kraus3(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus0(p) , depolarising_kraus3(p))).T)
                ) +
                (
                    kron(depolarising_kraus1(p) , depolarising_kraus0(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus1(p) , depolarising_kraus0(p))).T)
                ) +
                (
                    kron(depolarising_kraus1(p) , depolarising_kraus1(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus1(p) , depolarising_kraus1(p))).T)
                ) +
                (
                    kron(depolarising_kraus1(p) , depolarising_kraus2(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus1(p) , depolarising_kraus2(p))).T)
                ) +
                (
                    kron(depolarising_kraus1(p) , depolarising_kraus3(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus1(p) , depolarising_kraus3(p))).T)
                ) +
                (
                    kron(depolarising_kraus2(p) , depolarising_kraus0(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus2(p) , depolarising_kraus0(p))).T)
                ) +
                (
                    kron(depolarising_kraus2(p) , depolarising_kraus1(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus2(p) , depolarising_kraus1(p))).T)
                ) +
                (
                    kron(depolarising_kraus2(p) , depolarising_kraus2(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus2(p) , depolarising_kraus2(p))).T)
                ) +
                (
                    kron(depolarising_kraus2(p) , depolarising_kraus3(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus2(p) , depolarising_kraus3(p))).T)
                ) + 
                (
                    kron(depolarising_kraus3(p) , depolarising_kraus0(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus3(p) , depolarising_kraus0(p))).T)
                ) +
                (
                    kron(depolarising_kraus3(p) , depolarising_kraus1(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus3(p) , depolarising_kraus1(p))).T)
                ) +
                (
                    kron(depolarising_kraus3(p) , depolarising_kraus2(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus3(p) , depolarising_kraus2(p))).T)
                ) +
                (
                    kron(depolarising_kraus3(p) , depolarising_kraus3(p)) @
                    directEDstate @
                    numpy.conjugate((kron(depolarising_kraus3(p) , depolarising_kraus3(p))).T)
                )
            )
           )




# plot negativity as a function of the strength of the channel
# plot negativity as a function of the strength of the channel
#Defining partial_transpose function
def partial_transpose(input_matrix, dim_1, dim_2, subsystem):
    assert input_matrix.shape == (4, 4)
    assert dim_1 == 2
    assert dim_2 == 2
    assert subsystem in (1, 2)
    
    im = input_matrix
    
    if subsystem == 1:
        return numpy.array([
            [im[0, 0], im[0, 1], im[2, 0], im[2, 1]],
            [im[1, 0], im[1, 1], im[3, 0], im[3, 1]],
            [im[0, 2], im[0, 3], im[2, 2], im[2, 3]],
            [im[1, 2], im[1, 3], im[3, 2], im[3, 3]],
        ])
    elif subsystem == 2:
        return numpy.array([
            [im[0, 0], im[1, 0], im[0, 2], im[1, 2]],
            [im[0, 1], im[1, 1], im[0, 3], im[1, 3]],
            [im[2, 0], im[3, 0], im[2, 2], im[3, 2]],
            [im[2, 1], im[3, 1], im[2, 3], im[3, 3]],
        ])
#I want to ensure that when I maximise the negativity over parameters gamma and sigma, that p (strength of the channel) still remains a parameter
#building the negativity function
def eigenvalues1(p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                depolarised_directED(p),
                2,
                2,
                1,
            )
        )
    )

#defining the negativity function
def negativity1(p):
    eigenvalues = eigenvalues1(p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)

ps1 = numpy.linspace(0, 1, 100)
negativities_1 = [negativity1(p) for p in ps1]  

write_out(ps1 , negativities_1 , "output_directED_NB040.dat")




# Cubitt protocol




#Step 1: Alice and Bob prepare a classically correlated, separable state rhoabc,
#where Alice holds qubits a and c, and Bob holds qubit b

#Defining the ket and bra vectors for state psi[k]
def psiket(k):
    return 1/math.sqrt(2)*(zero+cmath.exp(1j*k*math.pi/2)*one)
def psibra(k):
    return 1/math.sqrt(2)*(zero.T+cmath.exp(-1j*k*math.pi/2)*one.T)
rho=1/6*(kron(psiket(0)@psibra(0),psiket(0)@psibra(0),zero@numpy.conjugate(zero.T))+
         kron(psiket(1)@psibra(1),psiket(-1)@psibra(-1),zero@numpy.conjugate(zero.T))+
         kron(psiket(2)@psibra(2),psiket(-2)@psibra(-2),zero@numpy.conjugate(zero.T))+
         kron(psiket(3)@psibra(3),psiket(-3)@psibra(-3),zero@numpy.conjugate(zero.T))
         )+1/6*(kron(zero@numpy.conjugate(zero.T),zero@numpy.conjugate(zero.T),one@numpy.conjugate(one.T))+
          kron(one@numpy.conjugate(one.T),one@numpy.conjugate(one.T),one@numpy.conjugate(one.T)))
             
                
                
#Step 2: CNOT on qubits a and c where a is control and c is target
CNOTac=kron(zero@numpy.conjugate(zero.T),numpy.identity(4))+kron(one@numpy.conjugate(one.T),numpy.identity(2),pauliX)
#applying the gate
applyingCNOTac=1/numpy.trace(CNOTac@rho@numpy.conjugate(CNOTac.T))*CNOTac@rho@numpy.conjugate(CNOTac.T)


#INTERMEDIATE STEP
# Applying the depolarising channel
# this channel is applied as the carrier qubit is transferred from Alice's site to Bob's

# Applying the Kraus operators
def depolarising_channel_Cubitt(p):
    return (
        (
            kron(numpy.identity(4),depolarising_kraus0(p)) @ 
            applyingCNOTac @
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus0(p)).T)
        ) +
        (
            kron(numpy.identity(4),depolarising_kraus1(p)) @ 
            applyingCNOTac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus1(p)).T)
        ) + 
        (
            kron(numpy.identity(4),depolarising_kraus2(p)) @ 
            applyingCNOTac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus2(p)).T)
        )+ 
        (
            kron(numpy.identity(4),depolarising_kraus3(p)) @ 
            applyingCNOTac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus3(p)).T)
        )
           )




#Step 3: CNOT on qubits b and c where b is control and c is target
CNOTbc=kron(numpy.identity(2),zero@numpy.conjugate(zero.T),numpy.identity(2))+kron(numpy.identity(2),one@numpy.conjugate(one.T),pauliX)
#applying the gate
def applyingCNOTbc(p):
    return 1/numpy.trace(CNOTbc@depolarising_channel_Cubitt(p)@numpy.conjugate(CNOTbc.T))*CNOTbc@depolarising_channel_Cubitt(p)@numpy.conjugate(CNOTbc.T)

#applying a projective measurement on qubit c
#ket
def projector_ket(gamma,sigma):
    return numpy.array([[numpy.cos(gamma)],
                        [math.e** (complex(0.0, sigma)) *numpy.sin(gamma)]])
#bra
def projector_bra(gamma,sigma):
    return numpy.conjugate(projector_ket(gamma,sigma)).T
#outer product
def projector(gamma,sigma):
    return projector_ket(gamma,sigma) @projector_bra(gamma,sigma)

#applying the measurement on qubit c
def measurement_of_c(gamma,sigma,p):
    return (
        ( 
            1/numpy.trace(kron(numpy.identity(4),projector(gamma,sigma)) @
            applyingCNOTbc(p) @ 
            numpy.conjugate(kron(numpy.identity(4),projector(gamma,sigma)).T))
        )
            *
        ( 
            kron(numpy.identity(4),projector(gamma,sigma)) @
            applyingCNOTbc(p) @
            numpy.conjugate(kron(numpy.identity(4),projector(gamma,sigma)).T)
        )
           )
#tracing out qubit c
def postmeasurement_state_Cubitt(gamma,sigma,p):
    return dTraceSystem(measurement_of_c(gamma,sigma,p), [3], 2)



#Measuring the entanglement between qubits a and b


#I want to ensure that when I maximise the negativity over parameters gamma and sigma, that p (strength of the channel) still remains a parameter
#building the negativity function
def eigenvaluesCubitt(gamma, sigma, p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                postmeasurement_state_Cubitt(
                    gamma,
                    sigma,
                    p
                ),
                2,
                2,
                1,
            )
        )
    )

#defining the negativity function
def negativityCubitt(gamma,sigma,p):
    eigenvalues = eigenvaluesCubitt(gamma,sigma,p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)

#optimising negativity for all values of the parameters using differential_evolution
def maximizeCubitt(fun, *args, **kwargs):
    def neg_fun(x, *a, **kwa):
        return -1 * fun(x, *a, **kwa)
    
    return differential_evolution(neg_fun, *args, **kwargs)

angles = []
def negativityCubitt_array(x, p):
    gamma,sigma = x
    return negativityCubitt(gamma,sigma, p)

def optimise_gamma_sigma_Cubitt(p):
    optCubitt = maximizeCubitt(negativityCubitt_array, [(0, math.pi), (0, 0.5*math.pi)], args=[p])
    return optCubitt.x

ps_Cubitt = numpy.linspace(0, 1, 100)
gamma_sigmas_Cubitt = [optimise_gamma_sigma_Cubitt(p) for p in ps_Cubitt]
negativities_Cubitt = [negativityCubitt_array(g_s_Cubitt, p) for (p, g_s_Cubitt) in zip(ps_Cubitt, gamma_sigmas_Cubitt)]
write_out(ps_Cubitt , negativities_Cubitt , "output_Cubitt_NB040.dat")
angles.append(gamma_sigmas_Cubitt)

# running the protocol as described in the Cubitt paper, where we measure in the computational basis at the end of the protocol
# we begin with applyingCNOTbc
measurement0 = kron(numpy.identity(4) , zero @ numpy.conjugate(zero.T))
# measuring qubit C
def correctoutcome0(p):
    return 1/numpy.trace(measurement0 @ applyingCNOTbc(p) @ numpy.conjugate(measurement0.T)) * (measurement0 @ applyingCNOTbc(p) @ numpy.conjugate(measurement0.T))
# tracing out qubit C
def postmeasurementstate0(p):
    return dTraceSystem(correctoutcome0(p) , [3] , 2)
# finding the eigenvalues of our state
def eigenvaluesCubittoutcome0(p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(postmeasurementstate0(p) , 2 , 2 , 1)
                            )
                    )
# finding the negativity of the state
def negativityCubittoutcome0(p):
    eigenvalues = eigenvaluesCubittoutcome0(p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)

negativities_outcome0 = []
for p in ps_Cubitt:
    negativities_outcome0.append(negativityCubittoutcome0(p))

plt.plot(ps_Cubitt , negativities_outcome0)
plt.show()


# Fedrizzi protocol



#defining the initial state of the memories
alphaAB = numpy.array([[0.375, 0.0, 0.0, 0.125],
             [0.0, 0.125, 0.0, 0.0],
             [0.0, 0.0, 0.125, 0.0],
             [0.125, 0.0, 0.0, 0.375]])
# defining the initial state of the carrier
carrier_state = numpy.array([[0.5, -0.25],
                       [-0.25, 0.5]])
# defining the joint state of cavities and carrier
joint_state = kron(alphaAB , carrier_state)



# Encoding Operation
# The separable carrier interacts with Alice's qubit through a CZ gate
# qubit 1 acts as the control and qubit 3 acts as the target
CZac = kron(zero@numpy.conjugate(zero.T),numpy.identity(4))+kron(one@numpy.conjugate(one.T),numpy.identity(2),pauliZ)
# applying the gate
applying_CZac = (1/numpy.trace(CZac@joint_state@numpy.conjugate(CZac.T)))*(CZac@joint_state@numpy.conjugate(CZac.T))


# INTERMEDIATE STEP
# Applying the depolarising channel as Alice sends the separable carrier to Bob's site
# Applying the Kraus operators defined above
def depolarising_channel_Fedrizzi(p):
    return (
        (
            kron(numpy.identity(4),depolarising_kraus0(p)) @ 
            applying_CZac @
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus0(p)).T)
        ) +
        (
            kron(numpy.identity(4),depolarising_kraus1(p)) @ 
            applying_CZac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus1(p)).T)
        ) + 
        (
            kron(numpy.identity(4),depolarising_kraus2(p)) @ 
            applying_CZac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus2(p)).T)
        )+ 
        (
            kron(numpy.identity(4),depolarising_kraus3(p)) @ 
            applying_CZac @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus3(p)).T)
        )
           )



# Decoding Operation
# Bob applies a CZ on qubits 2 and 3 as the separable carrier reaches his site
# qubit 2 acts as the control and qubit 3 acts as the target
CZbc = kron(numpy.identity(2),zero@numpy.conjugate(zero.T),numpy.identity(2))+kron(numpy.identity(2),one@numpy.conjugate(one.T),pauliZ)
# applying the gate
def applying_CZbc(p):
    return (1/numpy.trace(CZbc@depolarising_channel_Fedrizzi(p)@numpy.conjugate(CZbc.T)))*(CZbc@depolarising_channel_Fedrizzi(p)@numpy.conjugate(CZbc.T))

#applying the measurement on qubit c
def measurement_of_qubitC(ed,al,p):
    return (
        ( 
            1/numpy.trace(kron(numpy.identity(4),projector(ed,al)) @
            applying_CZbc(p) @ 
            numpy.conjugate(kron(numpy.identity(4),projector(ed,al)).T))
        )
            *
        ( 
            kron(numpy.identity(4),projector(ed,al)) @
            applying_CZbc(p) @
            numpy.conjugate(kron(numpy.identity(4),projector(ed,al)).T)
        )
           )
#tracing out qubit c
def postmeasurement_state_Fed1(ed,al,p):
    return dTraceSystem(measurement_of_qubitC(ed,al,p), [3], 2)



#Measuring the entanglement between qubits a and b


#I want to ensure that when I maximise the negativity over parameters gamma and sigma, that p (strength of the channel) still remains a parameter
#building the negativity function
def eigenvalues_Fed1(ed , al , p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                postmeasurement_state_Fed1(
                    ed,
                    al,
                    p
                ),
                2,
                2,
                1,
            )
        )
    )

#defining the negativity function
def negativity_Fed1(ed,al,p):
    eigenvalues = eigenvalues_Fed1(ed,al,p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)

#optimising negativity for all values of the parameters using differential_evolution
def maximize_Fed1(fun, *args, **kwargs):
    def neg_fun_Fed1(x, *a, **kwa):
        return -1 * fun(x, *a, **kwa)
    
    return differential_evolution(neg_fun_Fed1, *args, **kwargs)

def negativity_Fed1_array(x, p):
    ed,al = x
    return negativity_Fed1(ed,al, p)

def optimise_gamma_sigma_Fed1(p):
    opt_Fed1 = maximize_Fed1(negativity_Fed1_array, [(0, math.pi), (0, 0.5*math.pi)], args=[p])
    return opt_Fed1.x

ps_Fed1 = numpy.linspace(0, 1, 100)
gamma_sigmas_Fed1 = [optimise_gamma_sigma_Fed1(p) for p in ps_Fed1]
negativities_Fed1 = [negativity_Fed1_array(g_s_Fed1, p) for (p, g_s_Fed1) in zip(ps_Fed1, gamma_sigmas_Fed1)]
write_out(ps_Fed1 , negativities_Fed1 , "output_Fed1_NB040.dat")




# 2nd Iteration of Fedrizzi Protocol




# We define the new starting state as the postmeasurement state from above: postmeasurement_state_Fed1
# It is with this state that we form the joint state of our entangled pair and our separable carrier
# once again, the carrier will remain in the same state


# Building joint_state2 with entangled pair and separable carrier
def joint_state2(ed , al , p):
    return kron(postmeasurement_state_Fed1(ed , al , p) , carrier_state)

# Commencing the 2nd iteration of the Fedrizzi protocol

# Applying the CNOT on qubits A and C
def CZac2(ed , al , p):
    return CZac @ joint_state2(ed , al , p) @ numpy.conjugate(CZac.T)


# INTERMEDIATE STEP
# Applying the depolarising channel as Alice sends the separable carrier to Bob's site
# Applying the Kraus operators defined above
def depolarising_channel_Fedrizzi_2nd_iteration(ed , al , p):
    return (
        (
            kron(numpy.identity(4),depolarising_kraus0(p)) @ 
            CZac2(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus0(p)).T)
        ) +
        (
            kron(numpy.identity(4),depolarising_kraus1(p)) @ 
            CZac2(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus1(p)).T)
        ) + 
        (
            kron(numpy.identity(4),depolarising_kraus2(p)) @ 
            CZac2(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus2(p)).T)
        )+ 
        (
            kron(numpy.identity(4),depolarising_kraus3(p)) @ 
            CZac2(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus3(p)).T)
        )
           )



# Applying the CNOT on qubits B and C
def CZbc2(ed , al , p):
    return CZbc @ depolarising_channel_Fedrizzi_2nd_iteration(ed , al , p) @ numpy.conjugate(CZbc.T)
# Measuring qubit C
def measurement_of_qubitC2(ed , al , p):
    return (
        ( 
            1/numpy.trace(kron(numpy.identity(4),projector(ed , al)) @
            CZbc2(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T))
        )
            *
        ( 
            kron(numpy.identity(4),projector(ed , al)) @
            CZbc2(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T)
        )
           )

# Tracing out qubit C
def postmeasurement_state_Fed2(ed , al , p):
    return dTraceSystem(measurement_of_qubitC2(ed , al , p) , [3] , 2)



#building the negativity function
def eigenvalues_Fed2(ed , al , p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                postmeasurement_state_Fed2(
                    ed,
                    al,
                    p
                ),
                2,
                2,
                1,
            )
        )
    )
#defining the negativity function
def negativity_Fed2(ed , al , p):
    eigenvalues = eigenvalues_Fed2(ed , al , p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)
#optimising negativity for all values of the parameters using differential_evolution
def maximize_Fed2(fun, *args, **kwargs):
    def neg_fun_Fed2(x,*a, **kwa):
        return -1 * fun(x,*a, **kwa)
    
    return differential_evolution(neg_fun_Fed2, *args, **kwargs)

def negativity_Fed2_array(x , p):
    ed , al = x
    return negativity_Fed2(ed , al , p)

def optimise_gamma_sigma_Fed2(p):
    opt_Fed2 = maximize_Fed2(negativity_Fed2_array, [(0, math.pi), (0, 0.5*math.pi)], args=[p])
    return opt_Fed2.x

ps_Fed2 = numpy.linspace(0, 1, 100)
gamma_sigmas_Fed2 = [optimise_gamma_sigma_Fed2(p) for p in ps_Fed2]
negativities_Fed2 = [negativity_Fed2_array(g_s_Fed2, p) for (p , g_s_Fed2) in zip(ps_Fed2 , gamma_sigmas_Fed2)]
write_out(ps_Fed2 , negativities_Fed2 , "output_Fed2_NB040.dat")




# 3rd Iteration of Fedrizzi


# We define the new starting state as the postmeasurement state from above: postmeasurement_state3
# It is with this state that we form the joint state of our entangled pair and our separable carrier
# once again, the carrier will remain in the same state


# Building joint_state2 with entangled pair and separable carrier
def joint_state3(ed , al , p):
    return kron(postmeasurement_state_Fed2(ed , al , p) , carrier_state)


# Applying the CNOT on qubits A and C
def CZac3(ed , al , p):
    return CZac @ joint_state3(ed , al , p) @ numpy.conjugate(CZac.T)

# INTERMEDIATE STEP
# Applying the depolarising channel as Alice sends the separable carrier to Bob's site
# Applying the Kraus operators defined above
def depolarising_channel_Fedrizzi_3rd_iteration(ed , al , p):
    return (
        (
            kron(numpy.identity(4),depolarising_kraus0(p)) @ 
            CZac3(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus0(p)).T)
        ) +
        (
            kron(numpy.identity(4),depolarising_kraus1(p)) @ 
            CZac3(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus1(p)).T)
        ) + 
        (
            kron(numpy.identity(4),depolarising_kraus2(p)) @ 
            CZac3(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus2(p)).T)
        )+ 
        (
            kron(numpy.identity(4),depolarising_kraus3(p)) @ 
            CZac3(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus3(p)).T)
        )
           )

# Applying the CNOT on qubits B and C
def CZbc3(ed , al , p):
    return CZbc @ depolarising_channel_Fedrizzi_3rd_iteration(ed , al , p) @ numpy.conjugate(CZbc.T)
# Measuring qubit C
def measurement_of_qubitC3(ed , al , p):
    return (
        ( 
            1/numpy.trace(kron(numpy.identity(4),projector(ed , al)) @
            CZbc3(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T))
        )
            *
        ( 
            kron(numpy.identity(4),projector(ed , al)) @
            CZbc3(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T)
        )
           )

# Tracing out qubit C
def postmeasurement_state_Fed3(ed , al , p):
    return dTraceSystem(measurement_of_qubitC3(ed , al , p) , [3] , 2)



#building the negativity function
def eigenvalues_Fed3(ed , al , p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                postmeasurement_state_Fed3(
                    ed,
                    al,
                    p
                ),
                2,
                2,
                1,
            )
        )
    )
#defining the negativity function
def negativity_Fed3(ed , al , p):
    eigenvalues = eigenvalues_Fed3(ed , al , p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)
#optimising negativity for all values of the parameters using differential_evolution
def maximize_Fed3(fun, *args, **kwargs):
    def neg_fun_Fed3(x,*a, **kwa):
        return -1 * fun(x,*a, **kwa)
    
    return differential_evolution(neg_fun_Fed3, *args, **kwargs)

def negativity_Fed3_array(x , p):
    ed , al = x
    return negativity_Fed3(ed , al , p)

def optimise_gamma_sigma_Fed3(p):
    opt_Fed3 = maximize_Fed3(negativity_Fed3_array, [(0, math.pi), (0, 0.5*math.pi)], args=[p])
    return opt_Fed3.x

ps_Fed3 = numpy.linspace(0, 1, 100)
gamma_sigmas_Fed3 = [optimise_gamma_sigma_Fed3(p) for p in ps_Fed3]
negativities_Fed3 = [negativity_Fed3_array(g_s_Fed3, p) for (p , g_s_Fed3) in zip(ps_Fed3 , gamma_sigmas_Fed3)]
write_out(ps_Fed3 , negativities_Fed3 , "output_Fed3_NB040.dat")





# 4th Iteration of Fedrizzi




# We define the new starting state as the postmeasurement state from above: postmeasurement_state3
# It is with this state that we form the joint state of our entangled pair and our separable carrier
# once again, the carrier will remain in the same state


# Building joint_state2 with entangled pair and separable carrier
def joint_state4(ed , al , p):
    return kron(postmeasurement_state_Fed3(ed , al , p) , carrier_state)


# Commencing the 4th iteration of the Fedrizzi protocol

# Applying the CNOT on qubits A and C
def CZac4(ed , al , p):
    return CZac @ joint_state4(ed , al , p) @ numpy.conjugate(CZac.T)

# INTERMEDIATE STEP
# Applying the depolarising channel as Alice sends the separable carrier to Bob's site
# Applying the Kraus operators defined above
def depolarising_channel_Fedrizzi_4th_iteration(ed , al , p):
    return (
        (
            kron(numpy.identity(4),depolarising_kraus0(p)) @ 
            CZac4(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus0(p)).T)
        ) +
        (
            kron(numpy.identity(4),depolarising_kraus1(p)) @ 
            CZac4(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus1(p)).T)
        ) + 
        (
            kron(numpy.identity(4),depolarising_kraus2(p)) @ 
            CZac4(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus2(p)).T)
        )+ 
        (
            kron(numpy.identity(4),depolarising_kraus3(p)) @ 
            CZac4(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),depolarising_kraus3(p)).T)
        )
           )

# Applying the CNOT on qubits B and C
def CZbc4(ed , al , p):
    return CZbc @ depolarising_channel_Fedrizzi_4th_iteration(ed , al , p) @ numpy.conjugate(CZbc.T)
# Measuring qubit C
def measurement_of_qubitC4(ed , al , p):
    return (
        ( 
            1/numpy.trace(kron(numpy.identity(4),projector(ed , al)) @
            CZbc4(ed , al , p) @ 
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T))
        )
            *
        ( 
            kron(numpy.identity(4),projector(ed , al)) @
            CZbc4(ed , al , p) @
            numpy.conjugate(kron(numpy.identity(4),projector(ed , al)).T)
        )
           )

# Tracing out qubit C
def postmeasurement_state_Fed4(ed , al , p):
    return dTraceSystem(measurement_of_qubitC4(ed , al , p) , [3] , 2)



#building the negativity function
def eigenvalues_Fed4(ed , al , p):
    return numpy.real(
        numpy.linalg.eigvals(
            partial_transpose(
                postmeasurement_state_Fed4(
                    ed,
                    al,
                    p
                ),
                2,
                2,
                1,
            )
        )
    )
#defining the negativity function
def negativity_Fed4(ed , al , p):
    eigenvalues = eigenvalues_Fed4(ed , al , p)
    return numpy.sum((numpy.abs(eigenvalues) - eigenvalues) / 2)
#optimising negativity for all values of the parameters using differential_evolution
def maximize_Fed4(fun, *args, **kwargs):
    def neg_fun_Fed4(x,*a, **kwa):
        return -1 * fun(x,*a, **kwa)
    
    return differential_evolution(neg_fun_Fed4, *args, **kwargs)

def negativity_Fed4_array(x , p):
    ed , al = x
    return negativity_Fed4(ed , al , p)

def optimise_gamma_sigma_Fed4(p):
    opt_Fed4 = maximize_Fed4(negativity_Fed4_array, [(0, math.pi), (0, 0.5*math.pi)], args=[p])
    return opt_Fed4.x

ps_Fed4 = numpy.linspace(0, 1, 100)
gamma_sigmas_Fed4 = [optimise_gamma_sigma_Fed4(p) for p in ps_Fed4]
negativities_Fed4 = [negativity_Fed4_array(g_s_Fed4, p) for (p , g_s_Fed4) in zip(ps_Fed4 , gamma_sigmas_Fed4)]
write_out(ps_Fed4 , negativities_Fed4 , "output_Fed4_NB040.dat")