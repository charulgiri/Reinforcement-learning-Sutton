import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import seaborn as sns
import os
import time
os.system('clear')

plays=1000  # steps for each run/game
total_runs=2000  # total games
episilon=[0, 0.1,0.01]

scoreArr_e1=np.zeros(plays)
scoreArr_e2=np.zeros(plays)
scoreArr_greedy=np.zeros(plays)
optimlArr_greedy=np.zeros(plays)
optimlArr_e1=np.zeros(plays)
optimlArr_e2=np.zeros(plays)

rSum = np.zeros(10)
kAction=np.zeros(10)
valEstimates=np.zeros(10)



def main():
    x=[10,2,3,4,5,10]
    print(np.argmax(x))
    action = np.where(x == np.argmax(x))[0]
    print(action)
    init(optimlArr_greedy, scoreArr_greedy, episilon[0], total_steps=plays)
    init(optimlArr_e1, scoreArr_e1, episilon[1], total_steps=plays)
    init(optimlArr_e2, scoreArr_e2, episilon[2], total_steps=plays)

def reset():
    global rSum
    global kAction
    global valEstimates
    rSum    = np.zeros(10)
    kAction = np.zeros(10)
    valEstimates = np.zeros(10)

def init(optimlArr, scoreArr, episilon, total_steps=1000):
    for run in range(total_runs):
        reset()
        print(f"__________________ Run: {run} __________________ ")
        q_star = random.normal(loc=0, scale=1, size=10)
        print(f"q* Distribution:\n{q_star}")
        run_bandit_game(q_star, optimlArr, scoreArr, episilon, total_steps)

def run_bandit_game(q_star, optimlArr, scoreArr=scoreArr_greedy, episilon=0, total_steps=1000):
    action = random.randint(0,9)
    At     = action
    optimal_action = np.argmax(q_star)
    for step in range(total_steps):
        reward           = random.normal(loc=q_star[At], scale=1)
        kAction[At]     += 1
        rSum[At]        += reward
        scoreArr[step]  += reward
        valEstimates[At] = rSum[At]/kAction[At]

        greedyProb = np.random.random()
        #Exploit 
        if greedyProb > episilon:             
            action = np.argmax(valEstimates)

        #Explore
        else:                                  
            action = np.random.choice(10)
        At = action
        if action == optimal_action:
            optimlArr[step] += 1

if __name__=="__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {end_time-start_time}")

    scoreArr_greedy = scoreArr_greedy/total_runs
    scoreArr_e1 = scoreArr_e1/total_runs
    scoreArr_e2 = scoreArr_e2/total_runs
    optimlArr_greedy = optimlArr_greedy/total_runs
    optimlArr_e1 = optimlArr_e1/total_runs
    optimlArr_e2 = optimlArr_e2/total_runs

    agents=[ "Greedy", f"Ep {episilon[1]}", f"Ep {episilon[2]}"]

    scores=np.array([scoreArr_greedy, scoreArr_e1, scoreArr_e2])
    scores=np.transpose(scores)
    optimals=np.array([optimlArr_greedy, optimlArr_e1, optimlArr_e2])
    optimals=np.transpose(optimals)

    #Graph 1 - Averate rewards over all plays
    plt.title("10-Armed TestBed - Average Rewards")
    plt.plot(scores)
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()

    # Graph 2 - optimal selections over all plays
    plt.title("10-Armed TestBed - % Optimal Action")
    plt.plot(optimals * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(agents, loc=4)
    plt.show()







