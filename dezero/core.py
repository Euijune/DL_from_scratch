import numpy as np
import contextlib
import weakref

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    __array_priority__ = 200    # 연산자 우선순위 설정 (Ex. np.array의 __add__보다 Variable의 __add__함수가 먼저 호출됨)
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다. step 09를 참조하세요.'.format(type(data)))
        
        self.data = data
        self.name = name
        self.grad = None    # gradient = 기울기
        self.creator = None
        self.generation = 0 # 세대 수를 기록하는 변수

    def __len__(self):
        return len(self.data)

    def __repr__(self): # Variable 클래스의 인스턴스를 print하는 방식 지정
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # 세대를 기록한다(부모 세대 + 1)
    
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        
        funcs = []
        seen_set = set()

        def add_func(f):    # 함수의 중복추가 방지. Ex : 141p 그림 16-4 의 0세대 square함수는 1세대의 두 square함수에 의해 두 번 추가된다. 이것을 방지
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()    # 함수를 가져온다
            gys = [output().grad for output in f.outputs]   # output.grad는 약하게 참조된 데이터에 접근할 수 없다.

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)  # 메인 backward
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx    # 덮어쓰기 연산 x.grad += gx를 사용하면 문제가 생긴다.

                    if x.creator is not None:   # 역전파가 끝나지 않았다면, 해당 함수를 추가한다.
                        add_func(x.creator)

            if not retain_grad: # 말단 변수(x0, x1 등) 이외에는 미분값을 유지하지 않는다.
                for y in f.outputs:
                    y().grad = None # y는 약한 참조(weakref), 이 코드가 실행되면 참조값 카운트가 0이되어 미분값 데이터가 메모리에서 삭제
    
    def cleargrad(self):
        self.grad = None

    @property   # x.shape()대신 x.shape로 마치 인스턴스 변수인 것처럼 메서드를 사용 가능.
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)    # with 블록안에서, name으로 지정한 Config 클래스의 속성이 value값으로 설정됨
    try:
        yield
    finally:
        setattr(Config, name, old_value)    # with 블록 바깥에서는, 원래 값인 old_value로 돌아감
        
class Function:
    def __call__(self, *inputs): # 가변 길이 입출력
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # list(xs) unpacking
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):    # 순전파
        raise NotImplementedError()
    
    def backward(self, gys):    
        raise NotImplementedError()

def no_grad():
    return using_config('enable_backprop', False)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0,x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy*x1, gy*x0
    
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0 / x1 ** 2)
        return gx0, gx1
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c
        
    def forward(self, x):
        y = x**self.c
        return y
    
    def backward(self, gy):
        x = self.inputs
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__mul__ = mul    # Variable * float
    Variable.__add__ = add
    Variable.__rmul__ = mul    # float * Variable
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow