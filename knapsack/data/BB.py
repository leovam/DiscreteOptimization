import queue
class Node1:
    def __init__(self, level, profit, weight, bound):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.bound = bound
    
def bound(u, n ,W, items):
    if u.weight >= W:
        return 0
    profit_bound = u.profit
    j = u.level + 1
    totweight = u.weight
    
    while j < n and totweight + items[j].weight <= W:
        totweight += items[j].weight
        profit_bound += items[j].value
        j += 1
    if j < n:
        profit_bound += (W - totweight) * items[j].value / items[j].weight
    
    return profit_bound

def BB1(n, W, items):
    items = sorted(items, key=lambda k:float(k.value/k.weight), reverse=True)
    Q = queue.Queue()
    u = Node1(-1, 0, 0, 0)
    v = Node1(0, 0, 0, 0)
    Q.put(u)
    maxProfit = 0
    
    while not Q.empty():
        u = Q.get()
        
        if u.level == -1:
            v.level = 0
        
        if u.level == n - 1:
            continue
        v.level = u.level + 1
        
        v. weight = u.weight + items[v.level].weight
        v.profit = u.profit + items[v.level].value
        
        if v.weight <= W and v.profit > maxProfit:
            maxProfit = v.profit
        v.bound = bound(v, n, W, items)
        
        if v.bound > maxProfit:
            Q.put(v)
            
        v.weight = u.weight
        v.profit = u.profit
        v.bound = bound(v, n, W, items)
    
    return maxProfit