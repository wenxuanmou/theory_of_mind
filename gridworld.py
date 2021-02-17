import numpy as np
import pdb
from scipy.stats import dirichlet

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0,11):
        for j in range(0,11):
            if (state[i,j] == obj).all():
                return i,j
#finds an array in the "depth" dimension of the grid
# def findLoc_wall(state, obj):
#     ans=[]
#     for i in range(0,11):
#         for j in range(0,11):
#             if (state[i,j] == obj).all():
#                 ans.append([i,j])
#     return ans



#Initialize stationary grid, all items are placed deterministically
# def initGrid():
#     state = np.zeros((11,11,4))
#     #place player
#     state[0,1] = np.array([0,0,0,1])
#     #place wall
#     state[2,2] = np.array([0,0,1,0])
#     #place pit
#     state[1,1] = np.array([0,1,0,0])
#     #place goal
#     state[3,3] = np.array([1,0,0,0])
#     return state

#Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer(wall=True, n_obj=4):
    state = np.zeros((11,11,n_obj+1))
    #place player
    player = np.zeros((n_obj+1))
    
    player[-1] = 1
    palyer = player.astype(int)
    #print(player.astype(int),player.shape)
    a_loc = randPair(0,11)
    state[a_loc] = palyer#np.array([0,0,0,1])
    #print('a_loc ', a_loc)
    #place pit
    #state[7,7] = np.array([0,1,0,0])
    #place goal
    #state[5,5] = np.array([1,0,0,0])
    objs=[]
    objs_loc=[]
    for i in range(n_obj):
        obj = np.zeros((n_obj+1))
        obj[i] = 1
        obj=obj.astype(int)

        objs.append(obj)
        obj_loc = randPair(0,11)
        if obj_loc in objs_loc:
            continue
        objs_loc.append(obj_loc)
        state[obj_loc] = obj
        #print('gola ', i, ' position is ===== ', obj_loc)
        
    if a_loc in objs_loc:
        return initGridPlayer(wall, n_obj)
    if len(objs_loc)<n_obj:
        return initGridPlayer(wall, n_obj)



    goal1 = findLoc(state, np.array([1,0,0,0,0]))
    goal2 = findLoc(state, np.array([0,1,0,0,0]))
    goal3 = findLoc(state, np.array([0,0,1,0,0]))
    goal4 = findLoc(state, np.array([0,0,0,1,0]))

    if goal1 == None or goal2 == None or goal3 == None or goal4 == None:
        pdb.set_trace()
    
    return state

#Initialize grid so that goal, pit, wall, player are all randomly placed
# def initGridRand():
#     state = np.zeros((11,11,4))
#     #place player
#     state[randPair(0,11)] = np.array([0,0,0,1])
#     #place wall
#     state[randPair(0,11)] = np.array([0,0,1,0])
#     #place pit
#     state[randPair(0,11)] = np.array([0,1,0,0])
#     #place goal
#     state[randPair(0,11)] = np.array([1,0,0,0])
    
#     a = findLoc(state, np.array([0,0,0,1]))
#     w = findLoc_wall(state, np.array([0,0,1,0]))
#     g = findLoc(state, np.array([1,0,0,0]))
#     p = findLoc(state, np.array([0,1,0,0]))
#     #If any of the "objects" are superimposed, just call the function again to re-place
#     if (not a or not w or not g or not p):
#         #print('Invalid grid. Rebuilding..')
#         return initGridRand()
    
#     return state


def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,0,1]))
    #wall = findLoc_wall(state, np.array([0,0,1,0]))
    goal1 = findLoc(state, np.array([1,0,0,0,0]))
    goal2 = findLoc(state, np.array([0,1,0,0,0]))
    goal3 = findLoc(state, np.array([0,0,1,0,0]))
    goal4 = findLoc(state, np.array([0,0,0,1,0]))

    #pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((11,11,5))
    wall=[]

    if goal1 == None or goal2 == None or goal3 == None or goal4 == None:
        pdb.set_trace()
    state[goal1][0] = 1
    state[goal2][1] = 1
    state[goal3][2] = 1
    state[goal4][3] = 1

    actions = [[0,0],[-1,0],[1,0],[0,-1],[0,1]] # stay, up, down, left, right
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if (new_loc not in wall):
        if ((np.array(new_loc) <= (10,10)).all() and (np.array(new_loc) >= (0,0)).all()): # check if the new_loc is out of space
            #state[new_loc][4] = 1 # if it is not out of the space, we put player to the new_loc
            state[new_loc] = np.array([0,0,0,0,1])
    new_player_loc = findLoc(state, np.array([0,0,0,0,1]))
    if (not new_player_loc): # which means new_loc is out of space
        #pdb.set_trace() # so we keep player in the original position
        state[player_loc] = np.array([0,0,0,0,1])
    #re-place pit
    #state[pit][1] = 1
    #re-place wall
    #state[wall][2] = 1
    #re-place goal
    #pdb.set_trace()
    #print('gloal1, 2, 3, 4**** ', goal1, goal2, goal3, goal4)
    

    return state

def getLoc(state, level):
    for i in range(0,11):
        for j in range(0,11):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 4)
    #pit = getLoc(state, 1)
    goal1 = getLoc(state, 0)
    goal2 = getLoc(state, 1)
    goal3 = getLoc(state, 2)
    goal4 = getLoc(state, 3)


    if (player_loc in [goal1,goal2,goal3,goal4] ):
        return 100  # if it is in a goal position
    elif goal1==None or goal2 == None or goal3 == None or goal4 == None:
        return 100
    else:
        return -1 # if it is not in any goal position

    # if (player_loc == pit):
    #     return -10
    # elif (player_loc == goal):
    #     return 10
    # else:
    #     return -1

    
def dispGrid(state):
    #pdb.set_trace()

    grid = np.zeros((state.shape[0],state.shape[1]), dtype= str)
    player_loc = findLoc(state, np.array([0, 0, 0, 0, 1]))
    print(player_loc)
    #wall = findLoc_wall(state, np.array([0,0,1,0]))
    goals=[]
    goal = findLoc(state, np.array([1,0,0,0,0]))
    goals.append(goal)
    goal = findLoc(state, np.array([0,1,0,0,0]))
    goals.append(goal)
    goal = findLoc(state, np.array([0,0,1,0,0]))
    goals.append(goal)
    goal = findLoc(state, np.array([0,0,0,1,0]))
    goals.append(goal)
    #pdb.set_trace()

    #pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,11):
        for j in range(0,11):
            grid[i,j] = ' '
            
    if player_loc:
        grid[player_loc] = 'P' #player
    # if wall:
    #     grid[wall] = 'W' #wall
    
    # if pit:
    #     grid[pit] = '-' #pit
    for i, goal in enumerate(goals):
        if goal != None: # if the player cosume one goal, that goal will be None
            grid[goal[0],goal[1]] = str(i+1)
    # if None in goals:
    #      pdb.set_trace()
    # for i in range(len(goals)):
    #     grid[goals[i][0],goals[i][1]] =str(i+1)
#     for i in range(len(wall)):
#         grid[wall[i][0],wall[i][1]]='W'
    #pdb.set_trace()
    return grid


def get_policy(co_alpha,n_alpha,rand_seed):
    alpha = np.array([1, 1, 1,1,1])*co_alpha
    pi = dirichlet.rvs(alpha, size=1, random_state=rand_seed) # random_state=0 to set the random seed
    pi = pi[0] 

    return pi

def select_action(pi):
    id = np.argsort(pi)
    randn = np.random.uniform(0,1)
    if randn <= pi[id[0]]:
        action = id[0]
    elif pi[id[0]] < randn <= (pi[id[0]]+pi[id[1]]):
        action = id[1]
    elif (pi[id[0]]+pi[id[1]]) < randn <= (pi[id[0]]+pi[id[1]]+pi[id[2]]):
        action = id[2]
    elif (pi[id[0]]+pi[id[1]]+pi[id[2]]) < randn <= (pi[id[0]]+pi[id[1]]+pi[id[2]]+pi[id[3]]):
        action = id[3]
    else:
        action = id[4]
    return action

def action_spa(action):
    action_spa = np.zeros((11,11,5))

    if action == 0:
        action_spa[:,:,0]=1
    elif action == 1:
        action_spa[:,:,1]=1
    elif action == 2:
        action_spa[:,:,2]=1
    elif action == 3:
        action_spa[:,:,3]=1
    else:
        action_spa[:,:,4]=1

    return action_spa