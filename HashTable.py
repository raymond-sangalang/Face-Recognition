import sys
import re

class Node:
    def __init__(self, data = None):
        self.data = data
        self.prev = None
        self.next = None
    def set_data(self, data):
        self.data = data
    def get_data(self):
        return self.data
    def set_prev(self, prev):
        self.prev = prev
    def get_prev(self):
        return self.prev
    def has_prev(self):
        return self.prev is not None    
    def set_next(self, next):
        self.next = next
    def get_next(self):
        return self.next
    def has_next(self):
        return self.next is not None
    
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.nodeCount = 0
        
    def __del__(self):
        pCur = self.head
        while pCur is not None:
            pPre = pCur
            pCur = pCur.get_next()
            self.nodeCount -= 1
            del(pPre)
        
    def get_count(self):
        return self.nodeCount
        
    def prepend(self, newNode):
        self.insert_after(None, newNode)
        
    def append(self, newNode):
        self.insert_after(self.tail, newNode)
        
    def insert_after(self, curNode, newNode):
        # If list is empty
        if self.head is None:
            self.head = newNode
            self.tail = newNode
        # Insert after tail
        elif curNode is self.tail:
            self.tail.set_next(newNode)
            newNode.set_prev(self.tail)
            self.tail = newNode
        # Insert at head    
        elif curNode is None:
            newNode.set_next(self.head)
            self.head.set_prev(newNode)
            self.head = newNode
        #Insert somewhere in the middle
        else:
            sucNode = curNode.get_next()
            newNode.set_next(sucNode)
            newNode.set_prev(curNode)
            curNode.set_next(newNode)
            sucNode.set_prev(newNode)
        self.nodeCount += 1
        
    def remove(self, curNode):
        sucNode = curNode.get_next()
        predNode = curNode.get_prev()
        if sucNode is not None:
            sucNode.set_prev(predNode)
        if predNode is not None:
            predNode.set_next(sucNode)
        if curNode is self.head: #Removed head
            self.head = sucNode
        if curNode is self.tail: #Removed tail
            self.tail = predNode
        del(curNode)
        self.nodeCount -= 1
            
    def search(self, value):
        ptr = self.head
        while ptr is not None:
            if ptr.get_data() == value:
                return ptr
            ptr = ptr.get_next()
        return ptr
    
    def display(self):
        if self.head is None:
            return False
        ptr = self.head
        while ptr is not None:
            print(ptr.get_data())
            ptr = ptr.get_next()
    
    
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [LinkedList() for i in range(size)]
        self.collisions = 0
    
    def __len__(self):
        return len(self.table)
    
    def __del__(self):
        del(self.table)
        
    def insert(self, item):
        bucketList = self.table[self.get_hash(str(item))]
        if bucketList.get_count() > 0:
            self.collisions += 1
        newNode = Node(item)
        bucketList.prepend(newNode)
        
    def get_hash(self, key):
        return self.hash_remainder(key)
        #return self.hash_midSquare(key)
        #return self.hash_multiplicative(key)
    
    def hash_remainder(self, key):
        #choose table size to be a prime number not
        #too close to power of 2 or 10 like 997.        
        ASCII_sum = 0
        for i in range (len(key)):
            ASCII_sum += ord(key[i])
        return ASCII_sum % self.size
        
    def hash_midSquare(self, string):
        #This method works best if the table size is a power of two.
        #Square the number and take the middle part as the index.
        pattern = re.compile(r'([0-9]*)-?([0-9]*)')
        match = re.search(pattern, string)
        key = int(match.group(2))
        squaredKey = key * key
        R = 15
        lowBitsToRemove = (32 - R) / 2
        extractedBits = squaredKey >> int(lowBitsToRemove)
        extractedBits = extractedBits & (0xFFFFFFFF >> (32 - R))
        return extractedBits % self.size
        
    def hash_multiplicative(self, key):
        #Daniel J. Bernstein created a popular version of a multiplicative string hash function
        #that uses an initial value of 5381 and a multiplier of 33.
        #Bernstein's hash function performs well for hashing short English strings.
        stringHash = 5381
        hashMultiplier = 33
        for i in range(len(key)):
            stringHash = (stringHash * hashMultiplier) + ord(key[i])
        return abs(stringHash % self.size)
    
    def get_collisions(self):
        return self.collisions
    
    def search(self, key):
        bucketList = self.table[self.get_hash(key)]
        itemNode = bucketList.search(key)
        return itemNode.get_data() if itemNode is not None else None
    
    def remove(self, key):
        bucketList = self.table[self.get_hash(key)]
        itemNode = bucketList.search(key)
        if itemNode is None:
            return False
        bucketList.remove(itemNode)
        return True
    
    def display(self):
        [i.display() for i in self.table]
