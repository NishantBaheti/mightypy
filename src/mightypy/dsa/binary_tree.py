"""
Binary tree setup
"""

class Node:
    """A node for binary tree
    """

    def __init__(self, data):
        self.data = data
        self._left = None
        self._right = None

    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self,new_left):
        self._left = new_left

    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self,new_right):
        self._right = new_right

    def __repr__(self) -> str:
        return f"""Tree Node Value:{self.data}"""



class BinaryTree:

    def __init__(self) -> None:
        self._root = None   

    def _calc_height(self,node):
        if node is None:
            return 0
        else:
            # Compute the height of each subtree
            lheight = self._calc_height(node.left)
            rheight = self._calc_height(node.right)
    
            # Use the larger one
            return max(lheight, rheight) + 1

    @property
    def height(self):
        return self._calc_height(self._root)

    def insert(self,val):
        """
        Level order insertion.

        https://en.wikipedia.org/wiki/Breadth-first_search

        Args:
            val (Any): value of the node in tree.
        """
        if not self._root:
            self._root = Node(val)
            return

        q = []
        q.append(self._root)
        while len(q):
            temp = q.pop(0)

            if not temp.left:
                temp.left = Node(val)
                break
            else:
                q.append(temp.left)

            if not temp.right:
                temp.right = Node(val)
                break
            else:
                q.append(temp.right)

    def _level_order_traverse(self):
        if self._root is None:
            return
        
        q = []
        values = []
        q.append(self._root)
        while len(q):
            values.append(q[0].data)
            node = q.pop(0)

            if node.left is not None:
                q.append(node.left)
            
            if node.right is not None:
                q.append(node.right)
        return values

    def _inorder_traverse_rec(self, node, values=[]):
        """Inorder traversal recursive function

        Args:
            node (Node): tree node
            values (list, optional): list of values. Defaults to [].

        Returns:
            values (list): list of values
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            self._inorder_traverse_rec(left, values)
            values.append(data)
            self._inorder_traverse_rec(right, values)
        return values

    def _preorder_traverse_rec(self, node, values=[]):
        """preorder traversal recursive function

        Args:
            node (Node): tree node
            values (list, optional): list of values. Defaults to [].

        Returns:
            values (list): list of values
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            values.append(data)
            self._preorder_traverse_rec(left, values)
            self._preorder_traverse_rec(right, values)
        return values

    def _postorder_traverse_rec(self, node, values=[]):
        """Postorder traversal recursive function

        Args:
            node (Node): tree node
            values (list, optional): list of values. Defaults to [].

        Returns:
            values (list): list of values
        """
        if node:
            data = node.data
            left = node.left
            right = node.right

            self._postorder_traverse_rec(left, values)
            self._postorder_traverse_rec(right, values)
            values.append(data)
        return values

    def _inorder_traverse_stack(self):
        """Inorder traversal using stack

        Notes:
            Algorithm
                1. Create an empty stack S.
                2. Initialize current node as root
                3. Push the current node to S and set current = current->left until current is NULL
                4. If current is NULL and stack is not empty then 
                    a. Pop the top item from stack.
                    b. Print the popped item, set current = popped_item->right 
                    c. Go to step 3.
                5. If current is NULL and stack is empty then we are done.

        Returns:
            node_values (lsit): tree values
        """

        node_values = []
        current_node = self._root
        stack = []

        while True:
            if current_node is not None:
                stack.append(current_node)
                current_node = current_node.left

            elif stack:  # if stack is empty then true
                current_node = stack.pop()  # pop the node top of the stack
                node_values.append(current_node.data)
                current_node = current_node.right

            else:
                break
        return node_values

    def _preorder_traverse_stack(self):
        """Preorder Traversal using Stack

        Notes:
            Algorithm
                TODO

        Returns:
            node_values (list): node values
        """
        node_values = []
        current_node = self._root
        stack = []

        while True:
            if current_node is not None:
                node_values.append(current_node.data)
                stack.append(current_node)
                current_node = current_node.left

            elif stack:  # if stack is empty then true
                current_node = stack.pop()  # pop the node top of the stack
                current_node = current_node.right
            else:
                break
        return node_values

    def _postorder_traverse_stack(self):
        """Postorder Traversal using Stack

        Notes:
            Algorithm
                TODO
                
        Returns:
            node_values (list): node values
        """
        node_values = []
        current_node = self._root
        stack = []

        while True:
            if current_node is not None:
                stack.append(current_node)
                stack.append(current_node)
                current_node = current_node.left

            elif stack:  # if stack is empty then true
                current_node = stack.pop()  # pop the node top of the stack

                if len(stack) > 0 and stack[-1] == current_node:
                    current_node = current_node.right
                else:
                    node_values.append(current_node.data)
                    current_node = None
            else:
                break
        return node_values

    def traverse(self, order="in", method="stack"):
        """Tree traversal operation

        Args:
            order (str, optional): order in which the tree will be traversed. Defaults to "in".
                                    Options available -
                                    "level","in","pre","post"

        Raises:
            ValueError: If wrong order is passed. only "level","in","pre","post" is allowed

        Returns:
            values (list): Values of tree nodes in specified order
        """
        order = order.lower()
        method = method.lower()

        if order=="level":
            values = self._level_order_traverse()
        else:
            if method == "recursion":
                if order == "in":
                    values = self._inorder_traverse_rec(node=self._root, values=[])
                elif order == "pre":
                    values = self._preorder_traverse_rec(node=self._root, values=[])
                elif order == "post":
                    values = self._postorder_traverse_rec(node=self._root, values=[])
                else:
                    raise ValueError(f"order : '{order}' is not defined.")

            elif method == "stack":
                if order == "in":
                    values = self._inorder_traverse_stack()
                elif order == "pre":
                    values = self._preorder_traverse_stack()
                elif order == "post":
                    values = self._postorder_traverse_stack()
                else:
                    raise ValueError(f"order : '{order}' is not defined.")

            else:
                raise ValueError(f"method : '{method}' is not defined.")

        return values


if __name__ == "__main__":

    tree = BinaryTree()
    for i in range(1,6):
        tree.insert(i)


    print("height", tree.height)

    print("level", tree.traverse(order="level", method="stack"))


    print("post", tree.traverse(order="post",method="stack"))
    print("post", tree.traverse(order="post", method="recursion"))

    print("pre", tree.traverse(order="pre",method="stack"))
    print("pre", tree.traverse(order="pre", method="recursion"))

    print("in", tree.traverse(order="in", method="stack"))
    print("in", tree.traverse(order="in", method="recursion"))
