"""
Linked list setup
"""

class Node:
    def __init__(self, data):
        self.data = data 
        self._next = None

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self,new_next):
        self._next = new_next

    def __repr__(self):
        return f"Node Value : {self.data}"

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self,value):
        """Insert at the end.

        Args:
            value (object): Value to insert in linked list
        """
        temp = Node(value)
        if self.head is None:
            self.head = temp
        elif self.tail is None:
            self.tail = temp
            self.head.next = self.tail
        else:
            self.tail.next = temp
            self.tail = self.tail.next


    def push(self,value):
        """Push at the begining.

        Args:
            value (object): Value to push to list
        """
        temp = Node(value)
        if self.head is None:
            self.head = temp
        elif self.tail is None:
            self.tail = self.head
            self.head = temp
            self.head.next = self.tail
        else:
            temp.next = self.head
            self.head = temp

    def traverse(self):
        """Traverse the linked list.

        Returns:
            node_values (list) : linked list values
        """
        node_values = []
        current_node = self.head
        while current_node:
            node_values.append(current_node.data)
            current_node = current_node.next

        return node_values

    def __len__(self):

        size = 0
        curr_node = self.head

        while curr_node:
            size += 1
            curr_node = curr_node.next

        return size


if __name__ == "__main__":
    
    l_list = LinkedList()

    for i in range(10):
        l_list.push("push"+str(i))

    for i in range(10):
        l_list.append("insert"+str(i))


    print(l_list.traverse())

    print(len(l_list))
