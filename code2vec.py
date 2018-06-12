import ast
import numpy as np
import tensorflow as tf

class CodeSnippet:
    def __init__(self, file_path):
        source = open(file_path).read()
        self.tree = TreeNode(ast.parse(source))
        self.N = non_term_nodes(self.tree)
        self.T = term_nodes(self.tree)
        self.X = values(self.tree)
        self.s = self.tree
        self.tpairs = self._tpairs()
        self.rep = self._rep()
    def _tpairs(self):
        result = []
        for i in range(len(self.T)):
            for j in range(len(self.T)):
                if i != j:
                    result.append((self.T[i], self.T[j]))
        return result
    def _rep(self):
        return [path_context(start, end) for start, end in self.tpairs]
            
class Path(object):
    def __init__(self, start_node, end_node):
        self.values = create_path_values(start_node, end_node, None, 0)
        self.key = self.create_key()
    def create_key(self):
        result = ""
        for value in self.values:
            result += value
        return result
    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        return self.key == other.key
    def __hash__(self):
        return hash(self.key)
        

class TreeNode:
    def __init__(self, node, parent=None):
        self.parent = parent
        self.children = []
        self.value = map_to_value(node)
        self.value_type = value_type(node)
        self.expr = expr(node)
        if not is_term_node(self):
            for child in ast.iter_child_nodes(node):
                self.children.append(TreeNode(child, self))

def is_equal_path_context(p1, p2):
    if p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]:
        return True
    return False

def path_context(start_node, end_node):
    p = Path(start_node, end_node)
    return (start_node.value, p, end_node.value)
    

def path_down(start_node, end_node, num_direction_changes):
    for child in start_node.children:
        if child != start_node:
            p = create_path_values(child, end_node, "down", num_direction_changes)
            if p != None:
                p.insert(0, "__down__")
                p.insert(0, start_node.expr)
                return p
def path_up(start_node, end_node, num_direction_changes):
    p = create_path_values(start_node.parent, end_node, "up", num_direction_changes)
    if p != None:
        p.insert(0, "__up__")
        p.insert(0, start_node.expr)
        return p
    
def create_path_values(start_node, end_node, arrow, num_direction_changes):
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
    if is_term_node(tree):
        result.append(tree) 
    for child in tree.children:
        result.extend(term_nodes(child))
    return result


def non_term_nodes(tree):
    result = []
    if is_term_node(tree):
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


            


    
class BagOfPathPathContext:
    def __init__(self, snippets, tags, d):
        self.d = d
        self.tags = tags
        self.snippets = snippets
        self.all_paths = self._all_paths()
        self.all_path_contexts = self._all_path_contexts()
        self.value_vocab, self.values = self.create_value_vocab()
        self.word_vocab, self.words = self.create_word_vocab()
        self.path_vocab = self.create_path_vocab()
        self.tags_vocab = self.create_tags_vocab()
    def create_value_vocab(self):
        X = []
        for snippet in self.snippets:
            for term_node in snippet.T:
                X.append(term_node.value)
        X = list(set(X))
        return np.random.uniform(0, 1, (len(X), self.d)), X
    def create_tags_vocab(self):
        return np.random.uniform(0, 1, (len(self.tags), self.d))
    def create_path_vocab(self):
        return np.random.uniform(0, 1, (len(self.all_paths), self.d))
    def create_word_vocab(self):
        X = []
        for snippet in self.snippets:
            for term_node in snippet.T:
                if type(term_node.value) == str:
                    X.append(term_node.value)
        X = list(set(X))
        return np.random.uniform(0, 1, (len(X), self.d)), X
    def _all_paths(self):
        result = []
        for snippet in self.snippets:
            rep = snippet.rep
            paths = [p_context[1] for p_context in rep]
            result.extend(paths)
        return list(set(result))
    def _all_path_contexts(self):
        result = []
        for snippet in self.snippets:
            for p_context in snippet.rep:
                result.append(p_context)
        return result
    def embedding(self, p_context):
        index_start = self.words.index(p_context[0])
        index_end = self.words.index(p_context[2])
        for i, path in enumerate(self.all_paths):
            if path == p_context[1]:
                return [self.word_vocab[index_start], self.path_vocab[i], self.word_vocab[index_end]];
        return None


def attention_weight(c_tilde, a):
    denominator = sum([tf.reduce_sum([tf.exp(tf.matmul(c_tilde[j], a, transpose_a=True))]) for j in c_tilde.shape[0]])
    return [tf.exp(c_tilde[i], a, transpose_a=True) / denominator for i in range(c_tilde.shape[0])]

def code_vector(c_tilde, alpha):
    return tf.reduce_sum([tf.matmul(alpha[i], c_tilde[i]) for i in range(alpha.shape[0])])

def model_q(Y, v, tags_vocab):
    result = []
    denominator = tf.reduce_sum([tf.exp(tf.matmul(v, tags_vocab[j], transpose_a=True)) for j in range(tags_vocab.shape[0])])
    for i in range(tags_vocab.shape[0]):
        numerator = tf.exp(tf.matmul(v, tags_vocab.shape[0], transpose_a=True))
        result.append(numerator / denominator)
    return result


d = 3

snippets = [CodeSnippet("../hello.py"), CodeSnippet("../linear.py"), CodeSnippet("../lstm.py")]
Y = ["indexof", "getParameter"]

Bag = BagOfPathPathContext(snippets, Y, d)
c = [Bag.embedding(p_context) for p_context in Bag.all_path_contexts]
W = tf.Variable(tf.random_uniform([d, 3 * d]), name="W")
c_tilde = [tf.tanh(tf.matmul(W, c[i])) for i in range(len(c))]
a = tf.Variable(tf.random_uniform([d]), name="alpha")
alpha = attention_weight(c_tilde, a)
v = code_vector(alpha, c_tilde)
q = model_q(Y, v, Bag.tags_vocab)
    
print(embedding(path_context(snippets[0].T[0], snippets[0].T[1])))
