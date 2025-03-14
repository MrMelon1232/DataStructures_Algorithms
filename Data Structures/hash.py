"""
We're going to explore hash tables, hash functions, hash sets, hash maps
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Hashsets | note that sets have only unique items, therefore duplicates are not allowed
s = set()
print(s)

# Add item into set - O(1)
s.add(1)
s.add(2)
s.add(3)
print(s)

# Lookup item in set - O(1)
if 1 in s:
    print(True)

# Remove item in set - O(1)
s.remove(1)
print(s)

# Set construction of string - O(s) where s = length of string
string = "aaaaaabbbbbbcccccddd"
sett = set(string)
print(sett) # Should print {'d', 'c', 'b', 'a'}


###################################################################
# Hashmaps - Dictionaries in python
d = {'a': 1, 'b': 2, 'c': 3}
print(d)

# Adding key:val in dict - O(1)
d['d'] = 4
print(d)

# Checking key in dictionary - O(1)
if 'a' in d:
    print(True)

# Fetching value of key in dictionary - O(1)
print(d['a']) # Note: make sure that 'a' is in the dictionary or an error will occur (run the previous line by checking in dict)

# Loop over key:val pairs in dictionary - O(n)
for key, val in d.items():
    print(f'key {key}: val {val}')

# Removing item from dictionary - O(1)
d.pop('c')
print(d)

# If dictionary item does not exist and we want to increment the count for it the first time, use this
# in d.get(char, 0), the dictionary attemps to get the key 'char' and if its not found, it initializes its value to 0 and adds it to the dictionary
a = "hello"
for char in a:
    d[char] = 1 + d.get(char, 0)

# Function get(): retrieves a value in our map and if it doesn't exist, initialize it with an initial value
d.get('a', 0)


# Default dict | this library can be used to map lists to dictionaries, etc
from collections import defaultdict

default = defaultdict(list)
default[2]
print(default)

# Counter | makes a dictionary of items and the occurence of each item of the argument | dont use in interviews 
from collections import Counter

counter = Counter(string)
print(counter)