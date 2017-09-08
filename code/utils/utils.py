import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

def add(a, b):
    '''
    '''
    sum = a + b
    
    return sum


def divide(a, b):
    '''
    '''
    quotient = a / b
    
    return quotient


def multiply(a, b):
    '''
    '''
    product = a* b
    
    return product


def linspace(start, end, number_of_points):
    '''
    Linspace implementation using arrayfire.
    '''
    X = af.range(number_of_points, dtype = af.Dtype.f64)
    d = (end - start) / (number_of_points - 1)
    X = X * d
    X = X + start
    
    return X
