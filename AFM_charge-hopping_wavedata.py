# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:28:43 2018 (Finalized in Jun 2019)

@authors: Pavel Maly and Martin Ondracek

Issues: requires manual changes of parameters in the code,
        the formatting and output is quite crude
Numerical stability:
    possible trouble stem from
    the propagation of the diffusion equation
    -improved by Runge-Kutta with small substeps
    and suitable parameter regime
    -further improved by changing how the discreteness of the q coordinate is addressed
    when solving the diffusion equation.
    Instead of replacing the derivatives drho/dq and d2rho/dq2 by corresponding finite differences, as we did in older versions of this code,
    we used the following procedure: (i) We found the analytical kernel that solves the diffusion equation without the "hopping" terms (this is possible).
    (ii) We approximate the true kernel function (essentially a Gaussian function) in time dt by a "miniaturized" kernel with only a three-point support
    and (iii) perform the step in time from rho(t) to rho(t+dt) by computing the numerical convolution of rho(t) with this kernel.
    The hopping between rho1 and rho2 is then added separately, independent of this kernel based method.
Expect issues for: 
    higher temperature (tested up to 5K)
    fast dynamics in the potentials (large Lambda, hwrel)
    too fast hopping k_0 or too small reorganization energy (sharp resonance)
Solution: tweak timestep, subtimestep, dq and the q size
    watch out for too small q grid so that the probability leaks out!
Main thing to check: 
    convergence after number of periods - if not sure, increase tsize
"""

import numpy as np
np.set_printoptions(precision=2,suppress=True)

#import matplotlib.pyplot as plt

#for key-interrupting the convergence loop:)


Pi=3.14159265359

#grid for calculation, q=qleft +iq*dq, iq=[0,qsize], t=0+it*dt, it=[0,tsize]
#tsize to be set to include sufficient number of periods
#nper=tsize*dt/T0 and T0=1us
#currently set to 8 periods
qsize=200 #- check for numerical stability and leaking out of the grid
tsteps=80001
tsize=10001
dq=0.02
dt=0.0001
samples=100

subdtsteps=1 #substeps for numerical propagation
h=dt/subdtsteps
qleft=-2

RungeKutta=True

#kBT in pN A K^-1
#set 0.14*T[K]
kBT=0.14*5.0

#Dimensionless parameters for the potentials, not optimized
Lambda=2.00 #Inverse correlation time
hwrel=8.0 #Steepness with q
delta=1.0 #minima distance

#the densities
rho1=np.zeros((tsize,qsize))
rho2=np.zeros((tsize,qsize))

#generate first derivative field (not actually used in the present version)
def dqrhoq(rhoq):
    drho=np.zeros(qsize)
    for iq in range(1,qsize-1):
        drho[iq]=(rhoq[iq+1]-rhoq[iq-1])/2.0/dq
    drho[0]=(rhoq[1]-rhoq[0])/dq
    drho[qsize-1]=(rhoq[qsize-1]-rhoq[qsize-2])/dq
    return drho   

#generate second derivative field (not actually used in the present version)
def ddqrhoq(rhoq):
    ddrho=np.zeros(qsize)
    for iq in range(2,qsize-2):
        ddrho[iq]=(rhoq[iq+2]+rhoq[iq-2]-2.0*rhoq[iq])/4.0/dq/dq
    ddrho[1]=ddrho[2]
    ddrho[0]=ddrho[1]
    ddrho[qsize-2]=ddrho[qsize-3]
    ddrho[qsize-1]=ddrho[qsize-2]
    return ddrho

#multiply field by the coordinate q (not actually used in the present version)
def qrhoq(rhoq,d):
    q=qleft
    qrho=np.zeros(qsize)
    for iq in range(0,qsize):
        qrho[iq]=rhoq[iq]*(q+d)
        q+=dq
    return qrho

#the potentials
def V1(q):
    return 1/2*hwrel*(q-delta/2)*(q-delta/2)
def V2(q):
    return 1/2*hwrel*(q+delta/2)*(q+delta/2)

#the energy gap V2-V2+e0
def energygap(q,de0):
    return(V2(q)-V1(q)+de0)

def k12(q,de0):
    deltaE=energygap(q,de0)
    return k0*np.exp(-pow(deltaE-reorgE,2.0)/4/kBT/reorgE)

def k21(q,de0):
    deltaE=-energygap(q,de0)
    return k0*np.exp(-pow(deltaE-reorgE,2.0)/4/kBT/reorgE)

##backtransfer by detailed balance
#not so numerically stable for sharp resonance and tiny rate
#def k21(q,de0):
#    return(np.exp(-energygap(q,de0)/kBT)*k12(q,de0))

#arrays of rates k12(q), k21(q)
k12q=np.zeros(qsize)
k21q=np.zeros(qsize)

#setting rates for new energy gap
def setkq(de0):
    q=qleft
    for iq in range(0,qsize):
        k12q[iq]=k12(q,de0)
        k21q[iq]=k21(q,de0)
        q+=dq

drho_kern = np.zeros((2,3,qsize))
def genkern(tau):
    #generate the kernel elements that will be used to solve the diffusion equation in a stable way
    #(the procedure is based on a convolution of the density with a three-point kernel that approximates the resolvent)
    #argument tau = the time step times the relaxation rate
    
    #sigma2 = 2*kBT/hwrel*np.sinh(tau)*np.exp(-tau)
    sigma2 = 2*kBT/hwrel*tau/(dq*dq)
    drho_kerl = np.zeros((2,3,qsize))
    for iq in range(0,qsize):
        q = qleft + iq*dq
        for site in [0,1]:
            if(site==0):
                q0 = +delta/2
            else:
                q0 = -delta/2
            #mu = -2*(q - q0)*np.sinh(0.5*tau)*np.exp(-0.5*tau)/dq
            mu = -(q - q0)*tau/dq
            if(iq>0):
                drho_kern[site,-1,iq] = (sigma2 + mu*(mu-1)) / 2
            if(iq<qsize-1):
                drho_kern[site,+1,iq] = (sigma2 + mu*(mu+1)) / 2
            drho_kern[site,0,iq] = -(drho_kern[site,-1,iq] + drho_kern[site,+1,iq])

#TimeStep in the Smoluchovski equations
def rhostep(rho1q,rho2q,h):
    #dqrho1=dqrhoq(rho1q)
    #dqrho2=dqrhoq(rho2q)
    #res1=np.zeros(qsize)
    #res1=Lambda*(kBT/hwrel*ddqrhoq(rho1q)+(rho1q+qrhoq(dqrho1,-delta/2)))+k12q*rho2q-k21q*rho1q
    #res2=np.zeros(qsize)
    #res2=Lambda*(kBT/hwrel*ddqrhoq(rho2q)+(rho2q+qrhoq(dqrho2,+delta/2)))+k21q*rho1q-k12q*rho2q

    res1 = np.array(rho1q*drho_kern[0,0]) + np.roll(rho1q*drho_kern[0,1],+1) + np.roll(rho1q*drho_kern[0,-1],-1) + (k12q*rho2q-k21q*rho1q)*h
    res2 = np.array(rho2q*drho_kern[1,0]) + np.roll(rho2q*drho_kern[1,1],+1) + np.roll(rho2q*drho_kern[1,-1],-1) + (k21q*rho1q-k12q*rho2q)*h
    return (res1,res2)

#Electrostatics, parameters of the tip
alpha=0.25
k_el=-alpha*23070.7776
Rtip=3 #tip size
x0=[-5.2,5.2] #molecule positions
de_init=-100.0

#potential, in xz plane, Coulomb
def potential(x,z,n):
    #Bias-induced tip charge
    #return alpha/(z+Rtip)/pow(pow(x-x0[n],2)+z*z,0.5)
    
    #Image charge
    #return 0.5*k_el*Rtip/(pow(x-x0[n],2)+z*(2*Rtip+z))
    
    #Permanent tip charge
    return k_el/pow(pow(x-x0[n],2)+pow(z+Rtip,2),0.5)

#force, in xz plane, Coulombic
def force(x,z,n):
    #Bias-induced tip charge
    #return(potential(x,z,n)*(1/(z+Rtip)+z/(pow(x-x0[n],2)+z*z)))
    
    #Image charge
    #return k_el*Rtip*(z+Rtip)/pow(pow(x-x0[n],2)+z*(2*Rtip+z),2)
    
    #Permanent tip charge
    return k_el*(z+Rtip)/pow(pow(x-x0[n],2)+pow(z+Rtip,2),1.5)
    
V1arr=np.zeros(qsize)
V2arr=np.zeros(qsize)
#Iitial Conditions: 
#distribute Gaussian charge wavepackets over  molecules with given probability
def setIC(probabilities=None):
    q=qleft
    for iq in range(0,qsize):
        V1arr[iq]=V1(q)
        V2arr[iq]=V2(q)
        q+=dq
    q=qleft
    width=hwrel/2/kBT
    if probabilities is not None:
        print("Starting probabilities: "+str(probabilities[0])+", "+str(probabilities[1]))
        print("Initial wp inverse width: "+str(width))
        for iq in range(0,qsize):
            rho2[0,iq]=probabilities[1]*np.sqrt(width/Pi)*np.exp(-width*(q+delta/2)*(q+delta/2))
            rho1[0,iq]=probabilities[0]*np.sqrt(width/Pi)*np.exp(-width*(q-delta/2)*(q-delta/2))
            q+=dq
    else:
        q0=0 #center chare wavepackets at zero
        print("Starting point: "+str(q0))
        for iq in range(0,qsize):
            rho2[0,iq]=0.5*np.sqrt(width/Pi)*np.exp(-width*(q-q0)*(q-q0))
            rho1[0,iq]=0.5*np.sqrt(width/Pi)*np.exp(-width*(q-q0)*(q-q0))
            q+=dq

A=0.4 #oscillation amplitude

#time in microseconds
#Period of oscillations (in us, 1MHz)
T0=1

rate=70.71
reorgE=4.0
for Temperature in [1.2,5.0]:
    kBT=0.14*Temperature
    
    #The Marcus charge transfer rate
    #reorgE = reorganization energy
    k0=rate*np.sqrt(0.14*1.2/kBT) #rate prefactor, resonant rate in 1/us, including correct scaling with Temperature
    qaxis=np.arange(qsize)*dq+qleft
    genkern(Lambda*h)
    #for x in [3.00,6.00,9.00]:
    for x in [3.00]:
        
        fshift=np.array([])
        dissipation=np.array([])
        de0arr=np.array([])
        df_bkg=np.array([])
        
        #the x coordinate (chosen to intersect the isoenergetic curve)
        print("x coordinate: "+str(x))
        
        #The tip position
        z0=7.7 #z0 around which it oscillates
        
        #scanning the tip z position
        #now set for one position (zsize=1), for peak scanning leave zsize>1,
        #like zsize=111 set below and comment out zsize=1.
        zsize=111
        dz=0.05
        leftz=7.0
        zsize=1
        
        for iz in range(0,zsize):
            if zsize>1:
                z0=leftz+iz*dz
            print("\nz: "+str(z0))
            fcosint=0 #force cos integral over T0
            fsinint=0 #force sin integral over T0
            fcoslast=0 #force cos integral over the last T0
            fsinlast=0 #force sin integral over the lastT0

            fcosbkgint=0 #background force cos integral over T0
            fcosbkglast=0 #background force cos integral over the last T0

            #arrays for output
            dearr=np.array([])
            prob1=np.array([])
            prob2=np.array([])
            fcosarr=np.array([])
            fsinarr=np.array([])
            rate12arr=np.array([])
            rate21arr=np.array([])
            probabilities=[0.5,0.5]
            timeaxis=np.array([])
            de0=(-potential(x,z0,1)+potential(x,z0,0)+de_init)
            if(de0 > 100*kBT):
                probabilities[0]=1.0
                probabilities[1]=0.0
            elif(de0 < -100*kBT):
                probabilities[0]=0.0
                probabilities[1]=1.0
            else:
                Z=1+np.exp(-de0/kBT)
                probabilities[0]=1/Z
                probabilities[1]=np.exp(-de0/kBT)/Z
            setIC(probabilities=probabilities)
            
            #The time propagation
            rho1next=rho1[0]
            rho2next=rho2[0]
            for it in range(1,tsteps):
                t=it*dt
                deltaz=A*np.cos(2*Pi/T0*t)
                #energy gap for given tip position
                de0=(-potential(x,z0+deltaz,1)+potential(x,z0+deltaz,0)+de_init)
                setkq(de0)
                
                inirho=[rho1next,rho2next]
                
                if RungeKutta:
                    #Runge-Kutta (4th order)
                    for idt in range(0,subdtsteps):
                        res1=rhostep(inirho[0],inirho[1],h)
                        res2=rhostep(inirho[0]+res1[0]/2.0,inirho[1]+res1[1]/2.0,h)
                        res3=rhostep(inirho[0]+res2[0]/2.0,inirho[1]+res2[1]/2.0,h)
                        res4=rhostep(inirho[0]+res3[0],inirho[1]+res3[1],h)
                        step1=(res1[0]+2*res2[0]+2*res3[0]+res4[0])/6.0
                        step2=(res1[1]+2*res2[1]+2*res3[1]+res4[1])/6.0
                        rho1next=inirho[0]+step1
                        rho2next=inirho[1]+step2
                else:
                    #Euler
                    for idt in range(0,subdtsteps):
                        res=rhostep(inirho[0],inirho[1],h)
                        rho1next=inirho[0]+res[0]
                        rho2next=inirho[1]+res[1]
                
                #Probabilities for the two molecules
                p1=np.sum(rho1next)*dq
                p2=np.sum(rho2next)*dq

                #renormalize
                #rho1next[(rho1next < 0)] = 0
                #rho2next[(rho2next < 0)] = 0
                norm = p1 + p2
                #p1 = p1 / norm
                #p2 = p2 / norm
                rho1next = rho1next / norm
                rho2next = rho2next / norm

                forcenow=force(x,z0+deltaz,0)*p1+force(x,z0+deltaz,1)*p2
                fcos=forcenow*np.cos(2*Pi/T0*t)*dt
                fsin=forcenow*np.sin(2*Pi/T0*t)*dt
                fcosint+=fcos
                fsinint+=fsin
                fcosbkgint+=(force(x,z0+deltaz,0)*probabilities[0] + force(x,z0+deltaz,1)*probabilities[1])*np.cos(2*Pi/T0*t)*dt

                t_write = t - ((tsteps-1)*dt - T0)
                samplingT = T0 * 1.0 / samples
                if t_write >= 0 and ((t_write-0.9*dt)%samplingT >= (t_write+0.1*dt)%samplingT):
                #The condition after "and" above basically says: t_write is an integer multiple of samplingT within reasonable precision (+/- 10% of dt).
                    fcosarr=np.append(fcosarr,fcos)
                    fsinarr=np.append(fsinarr,fsin)
                    dearr=np.append(dearr,de0 / 1.602177)
                    prob1=np.append(prob1,p1)
                    prob2=np.append(prob2,p2)
                    if(p2>0):
                        rate12arr = np.append(rate12arr, np.sum(k12q*rho2next)*dq/p2)
                    else:
                        rate12arr=np.append(rate12arr,0.0)
                    if(p1>0):
                        rate21arr = np.append(rate21arr, np.sum(k21q*rho1next)*dq/p1)
                    else:
                        rate21arr=np.append(rate21arr,0.0)
                    np.savetxt("V1_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f_t%6.4f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0,t_write), np.column_stack(( qaxis , (V1arr-de0/2) / 1.602177)) , fmt = "%12.6f %15.8f")
                    np.savetxt("V2_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f_t%6.4f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0,t_write), np.column_stack(( qaxis , (V2arr+de0/2) / 1.602177)) , fmt = "%12.6f %15.8f")
                    np.savetxt("wave1_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f_t%6.4f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0,t_write), np.column_stack(( qaxis , rho1next)) , fmt = "%12.6f %15.8f")
                    np.savetxt("wave2_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f_t%6.4f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0,t_write), np.column_stack(( qaxis , rho2next)) , fmt = "%12.6f %15.8f")    
                    timeaxis=np.append(timeaxis,t_write)

                if it >= tsteps-tsize:
                    rho1[it-(tsteps-tsize)]=rho1next
                    rho2[it-(tsteps-tsize)]=rho2next

                #After each period, the dissipation and fshift are calculated, including the correct prefactors
                #if t % T0 == 0:
                if (t-dt)%T0 >= t%T0:
                    fcoslast=fcosint*0.01/A
                    fsinlast=-2*Pi*fsinint*A/1.602177 #dissipation in meV
                    print("Freq. shift: "+str(fcoslast))
                    print("En. dissipation: "+str(-fsinlast))
                    fcosint=0
                    fsinint=0
                    fcosbkglast=fcosbkgint*0.01/A
                    fcosbkgint=0
            
            fshift=np.append(fshift,fcoslast)
            dissipation=np.append(dissipation,fsinlast)
            de0arr=np.append(de0arr,(-potential(x,z0,1)+potential(x,z0,0)+de_init) / 1.602177)
            df_bkg=np.append(df_bkg,fcosbkglast)
            
        taxis=np.arange(tsize)*dt
        zaxis=np.arange(zsize)*dz+leftz+Rtip
        qaxis=np.arange(qsize)*dq+qleft
        tgrid,qgrid = np.meshgrid(taxis,qaxis,indexing='ij')
        
        if(zsize>=1):
            #np.savetxt("rho_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((np.ravel(tgrid),np.ravel(qgrid),np.ravel(rho1),np.ravel(rho2) )) , fmt = "%15.8f %15.8f %15.8f %15.8f")
            #np.savetxt("V1eq_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V1arr-potential(x,z0,0)-de_init/2)) , fmt = "%12.6f %15.8f")
            #np.savetxt("V2eq_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V2arr-potential(x,z0,1)+de_init/2)) , fmt = "%12.6f %15.8f")
            #np.savetxt("V1up_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V1arr-potential(x,z0+A,0)-de_init/2)) , fmt = "%12.6f %15.8f")
            #np.savetxt("V2up_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V2arr-potential(x,z0+A,1)+de_init/2)) , fmt = "%12.6f %15.8f")
            #np.savetxt("V1dn_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V1arr-potential(x,z0-A,0)-de_init/2)) , fmt = "%12.6f %15.8f")
            #np.savetxt("V2dn_A%3.1f_hw%04.1f_alpha%+4.2f_x%4.2f_z%4.2f.stab.dat"%(A,hwrel,alpha,x,z0), np.column_stack(( qaxis , V2arr-potential(x,z0-A,1)+de_init/2)) , fmt = "%12.6f %15.8f")
            
            np.savetxt("de_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((timeaxis,dearr)) , fmt = "%12.6f %15.8f")
            np.savetxt("occ1_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((timeaxis,prob1)) , fmt = "%12.6f %8.6f")
            np.savetxt("occ2_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((timeaxis,prob2)) , fmt = "%12.6f %8.6f")
            np.savetxt("rate1_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((timeaxis,rate12arr)) , fmt = "%12.6f %15.8f")
            np.savetxt("rate2_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f_z%4.2f.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x,z0), np.column_stack((timeaxis,rate21arr)) , fmt = "%12.6f %15.8f")
            #np.savetxt("kernel_T%3.1fK_hw%04.1f_KL%4.2f_dq%6.4f_dt%6.4f.dat"%(kBT/0.14,hwrel,Lambda,dq,h), np.column_stack((qaxis,drho_kern[0,-1],drho_kern[0,0],drho_kern[0,+1],drho_kern[1,-1],drho_kern[1,0],drho_kern[1,+1])) , fmt="%12.6f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f" )
        if(zsize>1):
            np.savetxt("z-df-exc_A%3.1f_T%3.1fK_hw%04.1f_KL%4.2f_lmb%05.2f_rate%05.1f_alpha%+4.2f_x%4.2f.stab.dat"%(A,kBT/0.14,hwrel,Lambda,reorgE,k0/np.sqrt(0.14*1.2/kBT),alpha,x), np.column_stack((zaxis,fshift,dissipation,df_bkg)))


"""
#Plotting usig pyplot
plt.plot(zaxis,fshift)
plt.ylabel("Frequency shift (Hz)")
plt.xlabel("z coordinate (Angstroms)")
plt.show()
plt.plot(zaxis,dissipation)
plt.ylabel("Energy dissipation (relative)")
plt.xlabel("z coordinate (Angstroms)")
plt.show()

plotdetails=False
if plotdetails:
    
    plt.plot(timeaxis,dearr)
    plt.ylabel("State energy difference")
    plt.xlabel("Time (microseconds)")
    plt.show()
    plt.plot(timeaxis,prob1)
    plt.plot(timeaxis,prob2)
    plt.plot(timeaxis,(prob1+prob2)/2)
    plt.ylabel("State probabilities")
    plt.xlabel("Time (microseconds)")
    plt.show()
    plt.plot(timeaxis,fcosarr)
    plt.ylabel("Cosine force")
    plt.xlabel("Time (microseconds)")
    plt.show()
    plt.plot(timeaxis,fsinarr)
    plt.ylabel("Sine force")
    plt.xlabel("Time (microseconds)")
    plt.show()
    wp1arr=np.zeros(qsize)
    wp2arr=np.zeros(qsize)
    for it in range(tsize-3-round(T0/dt),tsize-3):
        wp1arr+=rho1[it]
        wp2arr+=rho2[it]
    plt.plot(qaxis,wp1arr*dt)
    plt.plot(qaxis,wp2arr*dt)
    plt.ylabel("Average Wavepackets")
    plt.xlabel("q")
    plt.show()

    #the 2D plots have no units, just the grid
    #time runs top-down
    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.xlabel("q grid")
    plt.ylabel("t grid")
    plt.imshow(rho1,aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.xlabel("q grid")
    plt.ylabel("t grid")
    plt.imshow(rho2,aspect='auto')
    plt.colorbar()
    plt.show()
"""
