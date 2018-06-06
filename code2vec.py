import ast


class CodeSnippet:
    def __init__(self, file_path):
        source = open(file_path).read()
        self.tree = TreeNode(ast.parse(source))
        self.N = non_term_nodes(self.tree)
        self.T = term_nodes(self.tree)
        self.X = values(self.tree)
        self.s = self.tree

class TreeNode:
    def __init__(self, node, parent=None):
        self.parent = parent
        self.children = []
        self.value = map_to_value(node)
        self.value_type = value_type(node)
        self.expr = expr(node)
        if self.value_type == "non_term":
            for child in ast.iter_child_nodes(node):
                self.children.append(TreeNode(child, self))
def path_context(start_node, end_node):
    p = path(start_node, end_node, None, 0)
    return (start_node.value, p, end_node.value)
    

def path_down(start_node, end_node, num_direction_changes):
    for child in start_node.children:
        if child != start_node:
            p = path(child, end_node, "down", num_direction_changes)
            if p != None:
                p.insert(0, "__down__")
                p.insert(0, start_node.expr)
                return p
def path_up(start_node, end_node, num_direction_changes):
    p = path(start_node.parent, end_node, "up", num_direction_changes)
    if p != None:
        p.insert(0, "__up__")
        p.insert(0, start_node.expr)
        return p
    
def path(start_node, end_node, arrow, num_direction_changes):
    if start_node == end_node:
        return [end_node.expr]
    parent = None
    if start_node.parent != None:
        parent = start_node.parent
    if arrow == "up":
        if num_direction_changes == 0:
            p = path_down(start_node, end_node, num_direction_changes + 1)
            if p != None:
                return p
        if parent != None:
            p = path_up(start_node, end_node, num_direction_changes)
            if p != None:
                return p
    if arrow == "down":
        p = path_down(start_node, end_node, num_direction_changes)
        if p != None:
            return p
    else:
        if parent != None:
            p = path_up(start_node, end_node, num_direction_changes)
            if p != None:
                return p
        p = path_down(start_node, end_node, num_direction_changes)
        if p != None:
            return p
    return None



    
def value_type(node):
    if hasattr(node, "id"):
        return "id"
    if hasattr(node, "n"):
        return "n"
    if hasattr(node, "s"):
        return "s"
    else:
        return "non_term"
def is_term_node(node):
    if node.value_type == "non_term":
        return False
    return True

def len_tree(tree_node):
    result = 1
    for child in tree_node.children:
        result += len_tree(child)
    return result

def term_nodes(tree):
    result = []
    if tree.value_type != "non_term":
        result.append(tree) 
    for child in tree.children:
        result.extend(term_nodes(child))
    return result


def non_term_nodes(tree):
    result = []
    if tree.value_type != "non_term":
        return result
    else:
        result.append(tree)
    for child in tree.children:
        result.extend(non_term_nodes(child))
    return result

def values(tree):
    nodes = term_nodes(tree)
    return map_to_values(nodes)

def expr(node):
    return str(type(node)).replace("<class '_ast.", "").replace("'>", "")

def map_to_value(node):
    if hasattr(node, "n"):
        return node.n
    if hasattr(node, "s"):
        return node.s
    if hasattr(node, "id"):
        return node.id
    else:
        return str(type(node)).replace("<class '_ast.", "").replace("'>", "")
def map_to_values(nodes):
    return [map_to_value(node) for node in nodes]

snippet = CodeSnippet("../hello.py")

p = path_context(snippet.T[2], snippet.T[0])
print(p)
