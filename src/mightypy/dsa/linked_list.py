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
        self.next = new_next

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert(self,value):
        """Insert at the end

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
        """Push ar the start after head

        Args:
            value (object): Value to push to list
        """
        temp = Node(value)
        if self.head is None:
            self.head = temp
        elif self.tail is None:
            self.tail = temp
            self.head.next = self.tail
        else:
            temp.next = self.head.next
            self.head.next = temp

    def traverse(self):
        """Traverse the linked list

        Returns:
            node_values (list) : linked list values
        """
        node_values = []
        current_node = self.head
        while current_node:
            node_values.append(current_node.data)
            current_node = current_node.next

        return node_values


if __name__ == "__main__":
    
    l_list = LinkedList()

    for i in range(10):
        l_list.push("push"+str(i))

    for i in range(10):
        l_list.insert("insert"+str(i))


    print(l_list.traverse())
