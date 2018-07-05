# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:54:54 2018

@author: mihriban
"""
import numpy as np 
import random
import math
import numpy.matlib
from scipy.spatial import distance

def initialization(N,dim,up,down):
    X = np.random.uniform(down, up, (N, dim))
    return X
    

def Get_Functions_details(F):
    if F == 'F1':
        lb=-100
        ub=100
        dim=30
        
    elif F == 'F2':
        lb=-100
        ub=100
        dim=2
    
    elif F == 'F3':
        lb=-10
        ub=10
        dim=30
        
    elif F == 'F4':
        lb=-10
        ub=10
        dim=2
        
    elif F == 'F5':
        lb=-100
        ub=100
        dim=2
        
    elif F == 'F6':
        lb=-4.5
        ub=4.5
        dim=5
        
    elif F == 'F7':
        lb=-1.28
        ub=1.28
        dim=30
        
    elif F == 'F8':
        lb=-10
        ub=10
        dim=30
    
    elif F == 'F9':
        lb=-10
        ub=10
        dim=58
        
    elif F == 'F10':
        lb=-100
        ub=100
        dim=2
        
    elif F == 'F11':
        lb=-10
        ub=10
        dim=2
        
    elif F == 'F12':
        lb=-1.5
        ub=4
        dim=50
        
    elif F == 'F13':
        lb=-5
        ub=5
        dim=2
        
    elif F == 'F14':
        lb=-5
        ub=5
        dim=2
        
    elif F == 'F15':
        lb=0
        ub=math.pi
        dim=2


    return lb,ub,dim
    

    

def F1(x): #ackley_function
    return sum([i**2 for i in x])

def F2(x): #sphere
    return sum([i**2 for i in x])

def F3(x): #sum_squares_function
    return sum([(i+1)*x[i]**2 for i in range(len(x))])

def F4(x): #matyas_function
    return 0.26*F3(x) - 0.48*x[0]*x[1]

def F5(x): #easom_function
    return -math.cos(x[0])*math.cos(x[1])*math.exp(-(x[0] - math.pi)**2 - (x[1] - math.pi)**2)
    
def F6(x): #beale_function
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2

def F7(x): #quatic
    return sum((i*x[i]**4) for i in range(1, len(x)))+random.uniform(0,1)

def F8(x): #dixon price
      return (x[0] - 1)**2 + sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])


def F9(x): #cross_in_tray_function
    return round(-0.0001*(abs(math.sin(x[0])*math.sin(x[1])*math.exp(abs(100 -
                            math.sqrt(sum([i**2 for i in x]))/math.pi))) + 1)**0.1, 7)

def F10(x): #bohachevsky_function
    return x[0]**2 + 2*x[1]**2 - 0.3*math.cos(3*math.pi*x[0]) - 0.4*math.cos(4*math.pi*x[1]) + 0.7


def F11(x): #booth_function
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def F12(x): #mccormick_function
    return math.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1



def F13(x): #six_hump_camel_function
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1]\
           + (-4 + 4*x[1]**2)*x[1]**2


def F14(x): #three_hump_camel_function
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


def F15(x): #michalewicz_function
    return -sum([math.sin(x[i])*math.sin((i+1)*x[i]**2/math.pi)**20 for i in range(len(x))])

      
def S_func(r):
    f=0.5;
    l=1.5;
    o=f*np.exp(-r/l)-np.exp(-r) #Eq. (2.3)
    return o

def GOA(N, max_iter, lb,ub, dim):
    
    if np.size(ub)==1: 
        yeni_ub=np.ones((1,dim))
        ub = yeni_ub*ub
        yeni_lb=np.ones((1,dim))
        lb = yeni_lb*lb
    


    agents=initialization(N,dim,ub,lb);
    print(agents)
    #print(agents.shape)
    cMax=1
    cMin=4e-05

    #print("-----------------------")
    fitness = np.array([F5(x) for x in agents])
    #print(fitness)
    #print(fitness.shape)


    #print(np.size(fitness))
    sorted_indexes = np.argsort(fitness)
    #print("-----------------------")
    #print(sorted_indexes)
    #print("-----------------------")
    sorted_fitness = np.sort(fitness)
    #print(sorted_fitness)
    #print("-----------------------")
    sorted_agents = np.zeros((N,dim))
    for i in range(N):
        sorted_agents[i,:]= agents[(sorted_indexes[i]),:]
    #print(sorted_agents)
    #print(sorted_agents.shape)
    TargetFitness=sorted_fitness[0]
    TargetAgent=sorted_agents[0]
    #print(TargetFitness)
    #print(TargetAgent)
    
    agents_temp = np.ones((N,dim),dtype=object)
    for r in range(N):
        agents_temp[r,:] = agents[r,:]
    l = 2
    while l<max_iter:
        c = cMax - l * ((cMax-cMin)/max_iter)
        for i in range(N):
            s_i = np.zeros((N,dim),dtype=object)
            for j in range(N):
                if i != j:
                    mesafe=distance.euclidean(agents_temp[j,:] , agents_temp[i,:]);#
                    vektor = (agents_temp[j,:] - agents_temp[i,:])/mesafe
                    s_ij = ((ub-lb)*c / 2)*S_func(mesafe)*vektor
                    s_i = s_i + s_ij
                
            s_i_total = s_i
            x_new = c * s_i_total              
            agents_temp[i,:] = x_new[i]
    
        for r in range(N):
            agents[r,:] = agents_temp[r,:]
            for i in range(N):
                tp = agents[i,:]>ub
                tm = agents[i,:]<lb
                agents[i,:] = (agents[i,:]*(np.logical_not(tp+tm))+ub*tm)
        
                fitness = np.array([F5(x) for x in agents])
                sorted_indexes = np.argsort(fitness)
                sorted_fitness = np.sort(fitness)
                #print sorted_fitness
                #print sorted_fitness.shape
            
                for m in range(N):
                    sorted_agents[m,:]= agents[(sorted_indexes[m]),:]
            
                if sorted_fitness[0] < TargetFitness:
                    TargetAgent = sorted_agents[0]
                    TargetFitness = sorted_fitness[0]
            
                print "Iterasyon ",l
                print "Hedef ",TargetFitness
    
        l = l + 1;

    return TargetFitness,TargetAgent

def main():
    SearchAgents_no=30
    Function_name='F5'
    Max_iteration=500
    toplam=0;
    toplam2=np.ones((5,1),dtype=object)
    lb,ub,dim=Get_Functions_details(Function_name)
    #for i in range(30):
     #   Target_score,Target_pos = GOA(SearchAgents_no,Max_iteration,lb,ub,dim);
      #  toplam2[i]=Target_score
       # toplam +=Target_score
        #print(Target_score);
    #print("-------------------sssss")
    #sorted_score = np.sort(toplam2)
    #print sorted_score
    #print "average" ,toplam/30
    Target_score,Target_pos = GOA(SearchAgents_no,Max_iteration,lb,ub,dim);
    print("-------------------sssss")
    print(Target_pos);
    print(Target_score);
  
main()
