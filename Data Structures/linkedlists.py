"""
We're going to explore linked lists, singly and double
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# Singly linked lists:
class SinglyNode:
    
    # Initializing function (constructor)
    def __init__(self, val, next=None) -> None:
        self.val = val
        self.next = next
    
    # ToString method in python
    def __str__(self) -> str:
        return str(self.val)


# Creation of Singly Linked list
Head = SinglyNode(1)
A = SinglyNode(2)
B = SinglyNode(3)
C = SinglyNode(4)

Head.next = A
A.next = B
B.next = C

# Traverse List - O(n)
curr = Head
while curr:
    print(curr)
    curr = curr.next

# Display linked list - O(n)
def display(head):
    curr = head
    elements = []
    while curr:
        elements.append(str(curr.val))
        curr = curr.next
    print(' -> '.join(elements))    

display(Head) # Function displays linked list using join as such: 1 --> 2 --> ...

# Search for node value - O(n)
def search(head, val):
    curr = head
    while curr:
        if val == curr.val:
            return True
        curr = curr.next
    
    return False

print(search(Head, 3)) # Should print True



###################################################################
# Double linked list
class DoublyNode:
    def __init__(self, val, next=None, prev=None) -> None:
        self.val = val
        self.next = next
        self.prev = prev
    
    def __str__(self) -> str:
        return str(self.val)


# Doubly linked list creation
head = tail = DoublyNode(1)

# Display - O(n)
def display(head):
    curr = head
    elements = []
    while curr:
        elements.append(str(curr.val))
        curr = curr.next
    
    return (' <-> '.join(elements))

display(head)

# Insert at beginning - O(1)
def insert_at_beginning(head, tail, val):
    new_node = DoublyNode(val, next=head)
    head.prev = new_node
    return new_node, tail

head, tail = insert_at_beginning(head, tail, 3)
print(display(head))

# Insert at end - O(1)
def insert_at_end(head, tail, val):
    new_node = DoublyNode(val, prev=tail)
    tail.next = new_node
    return head, new_node

head, tail = insert_at_end(head, tail, 4)
print(display(head))


"""
Here is the following time complexity of singly and double linked lists:

Appending to end        | O(1) 
Popping from end        | O(1)
Insertion not from end  | O(1)
Deletion  not from end  | O(1)
Modifying element       | O(n)
Random access           | O(n)
Checking element exists | O(n)
"""
