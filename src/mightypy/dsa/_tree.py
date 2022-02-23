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

    @staticmethod
    def _calc_height(node):
        if node is None:
            return 0
        else:
            # Compute the height of each subtree
            lheight = BinaryTree._calc_height(node.left)
            rheight = BinaryTree._calc_height(node.right)
    
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

                    4.1. Pop the top item from stack
                    4.2. Print the popped item, set current = popped_item->right
                    4.3. Go to step 3
                    
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

    @staticmethod
    def _invert(node):

        node.left, node.right = node.right, node.left
        if node.left:
            BinaryTree._invert(node=node.left)
        if node.right:
            BinaryTree._invert(node=node.right)

    def invert(self):
        BinaryTree._invert(node=self._root)


    
class BST:

    def __init__(self) -> None:
        self._root = None

    @staticmethod
    def _insert(node, val):
        if node is None:
            return Node(val)
        else:
            if node.data == val:
                pass
            elif node.data > val:
                node.left = BST._insert(node=node.left,val=val)
            else:
                node.right = BST._insert(node=node.right,val=val)

        return node

    def insert(self, val):
        self._root = BST._insert(self._root,val)

    @staticmethod
    def _inorder_traverse(node, data_arr = []):
        if node is not None:

            BST._inorder_traverse(node.left, data_arr)
            data_arr.append(node.data)
            BST._inorder_traverse(node.right, data_arr)
            
        return data_arr

    @staticmethod
    def _preorder_traverse(node, data_arr = []):
        if node is not None:
            data_arr.append(node.data)
            BST._preorder_traverse(node.left, data_arr)
            BST._preorder_traverse(node.right, data_arr)
            
        return data_arr

    @staticmethod
    def _postorder_traverse(node, data_arr = []):
        if node is not None:
            BST._postorder_traverse(node.left, data_arr)
            BST._postorder_traverse(node.right, data_arr)
            data_arr.append(node.data)
            
        return data_arr

    

    @staticmethod
    def _level_order_traverse(node):
        if node is None:
            return []
        else:
            # actually this queue is working like a temp storage
            # this temp storage will store nodes that needs to be processed in FIFO
            # as we are talking about level order
            # so it needs to go by this order
            #                    
            #                         root
            #                      /       \
            #                 left1         right1
            #                /    \          /    \
            #             left2   right2   left3   right3
            # root -> left1 -> right1 -> left2 -> right2 -> left3 -> right3
            # 
            # queue's condition
            # [] <- root
            # [root]
            # root <-[ left1 right1 ]
            # left1 <- [ right1 left2 right2 ]
            # right1 <- [ left2 right2 left3 right3 ]
            # left2 <- [ right2 left3 right3 ]
            # right2 <- [ left3 right3 ]
            # left3 <- [ right3 ]
            # right3 <- []
        
            queue = []
            queue.append(node)

            # we need one data array to store traversed data no
            data_arr = []

            # now run the loop until we reach to an empty queue
            while len(queue):

                # take the first element from the queue and add it in data_arr
                # remove the element from queue but use it for left and right nodes iteration
                curr_node = queue.pop(0)

                data_arr.append(curr_node.data)
                
                if curr_node.left is not None: # adding left node to queue
                    queue.append(curr_node.left)
                
                if curr_node.right is not None: # adding right node to queue
                    queue.append(curr_node.right)
            
                # based on FIFO queue will process left node first 
                # then the right node.
            return data_arr


    def traverse(self, order="level"):
        
        order = order.lower()

        if order == "level":
            return BST._level_order_traverse(node=self._root)
        elif order == "in":
            return BST._inorder_traverse(node=self._root)
        elif order == "pre":
            return BST._preorder_traverse(node=self._root)
        elif order == "post":
            return BST._postorder_traverse(node=self._root)
        else:
            pass

if __name__ == "__main__":

    # tree = BinaryTree()
    # for i in range(1,7):
    #     tree.insert(i)


    # print("height", tree.height)

    # print("level", tree.traverse(order="level", method="stack"))

    # tree.invert()

    # print("level", tree.traverse(order="level", method="stack"))
    # print("height", tree.height)


    # print("post", tree.traverse(order="post",method="stack"))
    # print("post", tree.traverse(order="post", method="recursion"))

    # print("pre", tree.traverse(order="pre",method="stack"))
    # print("pre", tree.traverse(order="pre", method="recursion"))

    # print("in", tree.traverse(order="in", method="stack"))
    # print("in", tree.traverse(order="in", method="recursion"))


    bst = BST()

    for i in [3,4,6,12,9,11,5]:
        bst.insert(val=i)

    print(bst.traverse(order="in"))
    
