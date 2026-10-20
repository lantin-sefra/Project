"""
几个最简单的 Python 类，按 C++ 习惯对照看即可。

运行：python class_basics.py
"""


# ---------- 1. 最简类：只有构造 + 方法 ----------
# C++ 近似：
#   class Point {
#   public:
#     Point(double x, double y) : x_(x), y_(y) {}
#     double norm() const { return std::sqrt(x_*x_ + y_*y_); }
#   private:
#     double x_, y_;
#   };


class Point:
    def __init__(self, x, y):
        self.x = x  # 成员在这里“声明并赋值”
        self.y = y

    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5

    def __repr__(self):
        # 打印对象时好看一点，类似重载 operator<<
        return f"Point({self.x}, {self.y})"


# ---------- 2. 默认参数 + 可选成员 ----------
# C++: Counter(int start = 0) : n_(start) {}


class Counter:
    def __init__(self, start=0):
        self.n = start

    def inc(self, step=1):
        self.n += step
        return self.n


# ---------- 3. 组合：一个类里持有另一个对象 ----------
# C++: class Segment { Point a_, b_; ... };


class Segment:
    def __init__(self, a, b):
        self.a = a  # a, b 是 Point
        self.b = b

    def length(self):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        return (dx**2 + dy**2) ** 0.5


# ---------- 4. 继承 + super().__init__ ----------
# C++:
#   class Animal { public: Animal(std::string name) : name_(name) {} ...};
#   class Dog : public Animal {
#   public:
#     Dog(std::string name, std::string breed) : Animal(name), breed_(breed) {}
#   };


class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 先调基类构造
        self.breed = breed

    def speak(self):
        return "woof"


# ---------- 5. 和你 linreg_nn 同构：子模块挂到 self 上 ----------
# 这里不用 torch，只演示“构造里挂成员”的写法


class ToyNet:
    def __init__(self, in_dim, out_dim):
        # 真项目里这里会是：super().__init__(); self.linear = nn.Linear(...)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = [0.0] * (in_dim * out_dim)  # 假装参数

    def forward(self, x):
        # 演示用：不真算，只返回形状信息
        return f"forward: input_len={len(x)}, out_dim={self.out_dim}"

    def __call__(self, x):
        # 让 net(x) 等价于 net.forward(x)，类似 nn.Module 的用法
        return self.forward(x)





class Node:
    def __init__(self,data:int=0,next=None):
        self.data = data
        self.next = next

class LinkList:
    def __init__(self):
        node = Node()
        self.head = node

    def insertHead(self,node):
        node.next = self.head.next
        self.head.next = node
    
    def insertTail(self,node):
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = node
    
    def printList(self):
        current = self.head.next
        while current is not None:
            print(current.data, end=" ")
            current = current.next
        print()




# ---------- 跑一下 ----------
if __name__ == "__main__":
    p = Point(3, 4)
    print(p, "norm =", p.norm())

    c = Counter()
    print("counter:", c.inc(), c.inc(2))

    seg = Segment(Point(0, 0), Point(3, 4))
    print("segment length:", seg.length())

    d = Dog("Buddy", "corgi")
    print(d.name, d.breed, d.speak())

    net = ToyNet(2, 1)
    print(net([1.0, 2.0]))

    # with 需要 __enter__/__exit__；链表无外部资源，普通构造即可
    ll = LinkList()
    ll.insertHead(Node(1))
    ll.insertTail(Node(2))
    ll.printList()  # 1 2