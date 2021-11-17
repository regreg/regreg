from __future__ import print_function, division, absolute_import

from operator import add, mul
import warnings

import numpy as np
from scipy import sparse

def broadcast_first(a, b, op):
    """ apply binary operation `op`, broadcast `a` over axis 1 if necessary

    Parameters
    ----------
    a : ndarray
        If a is 1D shape (N,), convert to shape (N,1) before appling `op`.  This
        has the effect of making broadcasting happen over axis 1 rather than the
        default of axis 0.
    b : ndarray
        If a is 1D shape (P,), convert to shape (N,1) before appling `op`
    op : callable
        binary operation to apply to `a`, `b`

    Returns
    -------
    res : object
        shape equal to ``b.shape``
    """
    shape = b.shape
    if a.ndim == 1:
        a = a[:,None]
    if b.ndim == 1:
        b = b[:,None]
    return op(a, b).reshape(shape)


class AffineError(Exception):
    pass


class affine_transform(object):
    
    def __init__(self, linear_operator, affine_offset, diag=False, input_shape=None):
        """ Create affine transform

        Parameters
        ----------
        linear_operator : None or ndarray or sparse array or affine_transform
            Linear part of affine transform implemented as array or as
            affine_transform.  None results in no linear component.
        affine_offset : None or ndarray
            offset component of affine.  Only one of `linear_operator` and
            `affine_offset` can be None, because we need an input array to
            define the shape of the transform.
        diag : {False, True}, optional
            If True, interpret 1D `linear_operator` as the main diagonal of the
            a diagonal array, so that ``linear_operator =
            np.diag(linear_operator)``
        """
        # noneD - linear_operator is None
        # sparseD - linear_operator is sparse
        # affineD - linear_operator is an affine_transform
        # diagD - linear_operator is 1D representation of diagonal
        if linear_operator is None and affine_offset is None:
            raise AffineError('linear_operator and affine_offset cannot '
                              'both be None')

        if sparse.issparse(affine_offset):
            #Convert sparse offset to an array
            self.affine_offset = affine_offset.toarray().reshape(-1)
        else:
            self.affine_offset = affine_offset
        self.linear_operator = linear_operator

        if linear_operator is None:
            self.noneD = True
            self.sparseD = False
            self.affineD = False
            self.diagD = False
            self.input_shape = affine_offset.shape
            self.output_shape = affine_offset.shape
        else:
            self.noneD = False
            self.sparseD = sparse.isspmatrix(self.linear_operator)
            self.sparseD_csr = sparse.isspmatrix_csr(self.linear_operator)
            if self.sparseD and not self.sparseD_csr:
                warnings.warn("Linear operator matrix is sparse, but not csr_matrix. Convert to csr_matrix for faster multiplications!")
            if self.sparseD_csr:
                self.linear_operator_T = sparse.csr_matrix(self.linear_operator.T)


            # does it support the affine_transform API
            if np.alltrue([hasattr(self.linear_operator, n) for 
                           n in ['linear_map',
                                 'affine_map',
                                 'affine_offset',
                                 'adjoint_map',
                                 'input_shape',
                                 'output_shape']]):
                self.input_shape = self.linear_operator.input_shape
                self.output_shape = self.linear_operator.output_shape
                self.affineD = True
                self.diagD = False
            elif linear_operator.ndim == 1 and not diag:
                self.linear_operator = self.linear_operator.reshape((1,-1))
                self.diagD = False
                self.affineD = False
                self.input_shape = (self.linear_operator.shape[1],)
                self.output_shape = (1,)
            elif linear_operator.ndim == 1 and diag:
                self.diagD = True
                self.affineD = False
                self.input_shape = (linear_operator.shape[0],)
                self.output_shape = (linear_operator.shape[0],)
            elif (input_shape is not None) and (len(input_shape) == 2):
                #Input coefficients is a matrix
                self.input_shape = input_shape
                self.output_shape = (linear_operator.shape[0], input_shape[1])
                self.diagD = False
                self.affineD = False
            else:
                self.input_shape = (linear_operator.shape[1],)
                self.output_shape = (linear_operator.shape[0],)
                self.diagD = False
                self.affineD = False

    def linear_map(self, x):
        r"""Apply linear part of transform to `x`

        Return :math:`Dx`

        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D

        Returns
        -------
        Dx : ndarray
            `x` transformed with linear component

        Notes
        -----
        This routine is subclassed in affine_atom as a matrix multiplications,
        but could also call FFTs if D is a DFT matrix, in a subclass.
        """
        if self.noneD:
            return x
        elif self.affineD:
            return self.linear_operator.linear_map(x)
        elif self.sparseD:
            return self.linear_operator * x
        elif self.diagD:
            # Deal with 1D or 2D input or linear operator
            return broadcast_first(self.linear_operator, x, mul)
        return self.linear_operator.dot(x)

    def dot(self, x):
        r"""Apply linear part of transform to `x`.
        Returns `self.linear_map(x)` unless `x`
        is a transform, in which case it returns the composition.
        
        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D

        Returns
        -------
        Dx : ndarray
            `x` transformed with linear component

        """
        if not isinstance(x, affine_transform):
            return self.linear_map(x)
        else:
            return composition(self, x)
    
    @property
    def T(self, doc="Return the adjoint."):
        return adjoint(self)

    @property
    def ndim(self, doc="Dimension of array."):
        return len(self.input_shape) + len(self.output_shape)

    @property
    def shape(self, doc="Shape of linear map. " + 
              "Usual 2-tuple if transform can be represented by 2-dim array."):
        if len(self.input_shape) == 1 and len(self.output_shape) == 1:
            return (self.output_shape[0], self.input_shape[0])
        else:
            return (self.output_shape, self.input_shape)

    def affine_map(self, x):
        r"""Apply linear and affine offset to `x`

        Return :math:`Dx+\alpha`

        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D
        copy : {True, False}, optional
            If True, in situations where return is identical to `x`, ensure
            returned value is a copy.

        Returns
        -------
        Dx_a : ndarray
            `x` transformed with linear and offset components

        Notes
        -----
        This routine is subclassed in affine_atom as a matrix multiplications,
        but could also call FFTs if D is a DFT matrix, in a subclass.
        """
        if self.affineD:
            v = self.linear_operator.affine_map(x)
        else:
            v = self.linear_map(x)
        if self.affine_offset is not None:
            # Deal with 1D and 2D input, affine_offset cases
            return broadcast_first(self.affine_offset, v, add)
        return v

    def adjoint_map(self, u):
        r"""Apply transpose of linear component to `u`

        Return :math:`D^Tu`

        Parameters
        ----------
        u : ndarray
            array to which to apply transposed linear part of transform. Can be
            1D or 2D array
        copy : {True, False}, optional
            If True, in situations where return is identical to `u`, ensure
            returned value is a copy.

        Returns
        -------
        DTu : ndarray
            `u` transformed with transpose of linear component

        Notes
        -----
        This routine is currently a matrix multiplication, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        if self.noneD:
            return u
        if self.sparseD_csr:
            return self.linear_operator_T * u
        if self.sparseD:
            return self.linear_operator.T * u
        if self.diagD:
            # Deal with 1D or 2D input or linear operator
            return broadcast_first(self.linear_operator, u, mul)
        if self.affineD:
            return self.linear_operator.adjoint_map(u)
        return np.dot(self.linear_operator.T, u)


class linear_transform(affine_transform):
    """ A linear transform is an affine transform with no affine offset
    """
    def __init__(self, linear_operator, diag=False, input_shape=None):
        if linear_operator is None:
            raise AffineError('linear_operator cannot be None')
        affine_transform.__init__(self, linear_operator, None, diag=diag, input_shape=input_shape)


class selector(linear_transform):

    """
    Apply an affine transform after applying an
    indexing operation to the array.

    >>> from regreg.affine import selector, affine_transform
    >>> X = np.arange(30).reshape((6,5))
    >>> offset = np.arange(6)
    >>> transform = affine_transform(X, offset)
    >>> apply_to_first5 = selector(slice(0,5), (7,), transform)
    >>> apply_to_first5.linear_map(np.arange(7))
    array([ 30,  80, 130, 180, 230, 280])
    >>> np.dot(X, np.arange(5))
    array([ 30,  80, 130, 180, 230, 280])

    >>> apply_to_first5.affine_map(np.arange(7))
    array([ 30,  81, 132, 183, 234, 285])
    >>> np.dot(X, np.arange(5)) + offset
    array([ 30,  81, 132, 183, 234, 285])

    >>> result = np.array([275.,  290.,  305.,  320.,  335.,    0.,    0.])
    >>> np.testing.assert_allclose(result, apply_to_first5.adjoint_map(np.arange(6)))

    """

    def __init__(self,
                 index_obj,
                 initial_shape,
                 affine_transform=None,
                 reshape=None):

        self.index_obj = index_obj
        self.initial_shape = initial_shape
        self.reshape = reshape
        if affine_transform == None:
            test = np.empty(initial_shape)
            affine_transform = identity(test[index_obj].shape)
        self.affine_transform = affine_transform
        self.affine_offset = self.affine_transform.affine_offset
        self.input_shape = initial_shape
        if self.reshape is None:
            self.output_shape = self.affine_transform.output_shape
        else:
            self.output_shape = self.reshape
            
    def linear_map(self, x):
        x_indexed = x[self.index_obj]
        if self.reshape is not None:
            x_indexed.shape = self.reshape
        return self.affine_transform.linear_map(x_indexed)

    def affine_map(self, x):
        x_indexed = x[self.index_obj]
        if self.reshape is not None:
            x_indexed.shape = self.reshape
        return self.affine_transform.affine_map(x_indexed)

    def adjoint_map(self, u):
        if not hasattr(self, "_output"):
            self._output = np.zeros(self.initial_shape)
        if self.reshape is not None:
            u.shape = self.affine_transform.output_shape
        self._output[self.index_obj] = self.affine_transform.adjoint_map(u)
        return self._output

class scaler(linear_transform):

    def __init__(self, scalings):
        self.scalings = np.asarray(scalings)
        self.input_shape = self.output_shape = self.scalings.shape

    def linear_map(self, x):
        return self.scalings * x

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, u):
        return self.linear_map(u)

class reshape(linear_transform):

    """
    Reshape the output of an affine transform.

    """

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def linear_map(self, x):
        return x.reshape(self.output_shape)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, u):
        return u.reshape(self.input_shape)

# def tensor(T, first_input_index):
#     input_shape = T.shape[first_input_index:]
#     output_shape = T.shape[:first_input_index]

#     Tm = T.reshape((np.product(output_shape),
#                     np.product(input_shape)))
#     reshape_input = reshape(input_shape, Tm.shape[1])
#     reshape_output = reshape(Tm.shape[0], output_shape)
#     return composition(reshape_output, Tm, reshape_input)

def normalize(transform,
              center=True,
              scale=True,
              intercept=False,
              ddof=1,
              value=1):
    """
    Parameters
    ----------
    M : ndarray or scipy.sparse
        The matrix to be normalized. If an ndarray and inplace=True,
        then the values of M are modified in place. Sparse matrices
        are not modified in place.

    center : bool
        Center the columns?

    scale : bool
        Scale the columns?

    intercept : bool
        Include an intercept column?

    ddof : float
        Used to correct degrees of freedom in standard
        deviation calculation.

    value : float
        Set the std of the columns to be value.

    Returns
    -------

    final_map : linear_transform

    """

    M = transform # should be a matrix
    sparseM = sparse.isspmatrix(M)
    n = M.shape[0]

    if center:
        if not sparseM:
            col_means = M.mean(0)
        else:
            tmp = M.copy()
            col_means = np.asarray(tmp.mean(0)).reshape(-1)
            tmp.data **= 2
    else:
        col_means = np.zeros(M.shape[1])
        ddof = ddof - 1

    if scale:
        if not sparseM:
            col_scales = np.sqrt((np.sum(M**2,0) - n * col_means**2) / (n - ddof)) 
        else:
            col_scales = np.asarray(np.sqrt((np.asarray(tmp.sum(0)).reshape(-1) -
                                             n * col_means**2) / (n - ddof))).reshape(-1)
        nonzero_scales = col_scales > 0
        inv_col_scales = np.zeros_like(col_scales)
        inv_col_scales[nonzero_scales] = 1 / col_scales[nonzero_scales]

        diag_transform = linear_transform(inv_col_scales * value,
                                          diag=True)
        maps = [M, diag_transform]
    else:
        maps = [M]

    if center:
        maps = [centering((n,))] + maps

    final_map = composition(*maps)

    if intercept:
        return add_intercept(final_map)
    return final_map

class add_intercept(affine_transform):

    def __init__(self, transform):
        self._transform = astransform(transform)
        self.input_shape = (self._transform.input_shape[0] + 1,) # assumes a vector input, 
        self.output_shape = self._transform.output_shape
        self.linear_operator = None
        
    def affine_map(self, x): # first coordinate is intercept
        return self._transform.affine_map(x[1:]) + x[0]

    def linear_map(self, x):
        return self._transform.linear_map(x[1:]) + x[0]

    def adjoint_map(self, u):
        return np.hstack([u.sum(), self._transform.adjoint_map(u)])

class centering(affine_transform):

    def __init__(self, input_shape):
        self.input_shape = self.output_shape = input_shape
        self.affine_offset = None
        self.linear_operator = None

    def affine_map(self, x):
        return self.linear_map(x)

    def linear_map(self, x):
        return x - x.mean(0)

    def adjoint_map(self, x):
        return self.linear_map(x)
                 
class identity(affine_transform):

    """
    Identity transform
    """

    def __init__(self, input_shape):
        self.input_shape = self.output_shape = input_shape
        self.affine_offset = None
        self.linear_operator = None

    def affine_map(self, x):
        return self.linear_map(x)

    def linear_map(self, x):
        return x

    def adjoint_map(self, x):
        return self.linear_map(x)

class vstack(affine_transform):
    """
    Stack several affine transforms vertically together though
    not necessarily as a big matrix.
    """

    def __init__(self, transforms):
        self.input_shape = -1
        self.output_shapes = []
        self.transforms = []
        self.output_slices = []
        total_output = 0
        for transform in transforms:
            transform = astransform(transform)
            if self.input_shape == -1:
                self.input_shape = transform.input_shape
            else:
                if transform.input_shape != self.input_shape:
                    raise ValueError("input dimensions don't agree")
            self.transforms.append(transform)
            self.output_shapes.append(transform.output_shape)
            increment = np.product(transform.output_shape)
            self.output_slices.append(slice(total_output, total_output + increment))
            total_output += increment

        self.output_shape = (total_output,)
        self.group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                     for i, shape in enumerate(self.output_shapes)])
        self.output_groups = self.group_dtype.names 

        # figure out the affine offset
        self.affine_offset = np.empty(self.output_shape)
        x = np.zeros(self.input_shape)
        for g, t in zip(self.output_slices, self.transforms):
            self.affine_offset[g] = t.affine_map(x)
        if np.all(np.equal(self.affine_offset, 0)):
            self.affine_offset = None
            
    def linear_map(self, x):
        result = np.empty(self.output_shape)
        for g, t in zip(self.output_slices, self.transforms):
            result[g] = t.linear_map(x)
        return result

    def affine_map(self, x):
        result = np.empty(self.output_shape)
        for g, t in zip(self.output_slices, self.transforms):
            result[g] = t.linear_map(x)
        if self.affine_offset is not None:
            return result + self.affine_offset
        else:
            return result

    def adjoint_map(self, u):
        result = np.zeros(self.input_shape)
        for g, t, s in zip(self.output_slices, self.transforms,
                           self.output_shapes):
            result += t.adjoint_map(u[g].reshape(s))
        return result

class hstack(affine_transform):
    """
    Stack several affine transforms horizontally together though
    not necessarily as a big matrix.
   
    """

    def __init__(self, transforms):
        self.output_shape = -1
        self.input_shapes = []
        self.transforms = []
        self.input_slices = []
        total_input = 0
        for transform in transforms:
            transform = astransform(transform)
            if self.output_shape == -1:
                self.output_shape = transform.output_shape
            else:
                if transform.output_shape != self.output_shape:
                    raise ValueError("output dimensions don't agree")
            self.transforms.append(transform)
            self.input_shapes.append(transform.input_shape)
            increment = np.product(transform.input_shape)
            self.input_slices.append(slice(total_input, total_input + increment))
            total_input += increment

        self.input_shape = (total_input,)
        self.group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                     for i, shape in enumerate(self.input_shapes)])
        self.input_groups = self.group_dtype.names 

        # figure out the affine offset
        self.affine_offset = np.zeros(self.output_shape)
        for g, s, t in zip(self.input_slices, self.input_shapes,
                           self.transforms):
            self.affine_offset += t.affine_map(np.zeros(s))
        if np.all(np.equal(self.affine_offset, 0)):
            self.affine_offset = None

    def linear_map(self, x):
        result = np.zeros(self.output_shape)
        for g, t, s in zip(self.input_slices, self.transforms,
                           self.input_shapes):
            result += t.linear_map(x[g].reshape(s))
        return result

    def affine_map(self, x):
        result = np.zeros(self.output_shape)
        for g, t, s in zip(self.input_slices, self.transforms,
                        self.input_shapes):
            result += t.linear_map(x[g].reshape(s))
        if self.affine_offset is not None:
            return result + self.affine_offset
        else:
            return result

    def adjoint_map(self, u):
        result = np.empty(self.input_shape)
        #XXX this reshaping will fail for shapes that aren't
        # 1D, would have to view as self.group_dtype to
        # take advantange of different shapes
        for g, t, s in zip(self.input_slices, self.transforms,
                           self.input_shapes):
            result[g] = t.adjoint_map(u).reshape(-1)
        return result

class product(affine_transform):
    """
    Create a transform that maps the product of the inputs
    to the product of the outputs.
   
    """

    def __init__(self, transforms):
        self.output_shapes = []
        self.input_shapes = []
        self.transforms = []
        self.input_slices = []
        self.output_slices = []
        total_input = 0
        total_output = 0
        for transform in transforms:
            transform = astransform(transform)
            self.transforms.append(transform)
            self.input_shapes.append(transform.input_shape)
            self.output_shapes.append(transform.output_shape)
            input_increment = np.product(transform.input_shape)
            output_increment = np.product(transform.output_shape)
            self.input_slices.append(slice(total_input, total_input + input_increment))
            self.output_slices.append(slice(total_output, total_output + output_increment))
            total_input += input_increment
            total_output += output_increment

        self.input_shape = (total_input,)
        self.output_shape = (total_output,)
        self.input_group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                           for i, shape in enumerate(self.input_shapes)])
        self.input_groups = self.input_group_dtype.names 

        self.output_group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                           for i, shape in enumerate(self.input_shapes)])
        self.output_groups = self.output_group_dtype.names 

        # figure out the affine offset
        self.affine_offset = np.zeros(self.output_shape)
        for g, s, t in zip(self.output_slices, self.input_shapes,
                           self.transforms):
            self.affine_offset[g] = t.affine_map(np.zeros(s))
        if np.all(np.equal(self.affine_offset, 0)):
            self.affine_offset = None

    def linear_map(self, x):
        result = np.zeros(self.output_shape)
        for og, ig, t, s in zip(self.output_slices,
                                self.input_slices, 
                                self.transforms,
                                self.input_shapes):
            result[og] = t.linear_map(x[ig].reshape(s))
        return result

    def affine_map(self, x):
        result = np.zeros(self.output_shape)
        for og, ig, t, s in zip(self.output_slices,
                                self.input_slices, 
                                self.transforms,
                                self.input_shapes):
            result[og] = t.linear_map(x[ig].reshape(s))
        if self.affine_offset is not None:
            return result + self.affine_offset
        else:
            return result

    def adjoint_map(self, u):
        result = np.empty(self.input_shape)
        for og, ig, t, s in zip(self.output_slices,
                                self.input_slices, 
                                self.transforms,
                                self.input_shapes):
            result[ig] = t.adjoint_map(u[og]).reshape(-1)
        return result


def power_L(transform, max_its=500,tol=1e-8, debug=False):
    """
    Approximate the largest singular value (squared) of the linear part of
    a transform using power iterations
    
    TODO: should this be the largest singular value instead (i.e. not squared?)
    """

    transform = astransform(transform)
    v = np.random.standard_normal(transform.input_shape)
    old_norm = 0.
    norm = 1.
    itercount = 0
    while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
        v = transform.adjoint_map(transform.linear_map(v))
        old_norm = norm
        norm = np.linalg.norm(v)
        v /= norm
        if debug:
            print("L", norm)
        itercount += 1
    return norm

def astransform(X):
    """
    If X is an affine_transform, return X,
    else try to cast it as an affine_transform
    """
    if isinstance(X, affine_transform):
        return X
    else:
        return linear_transform(X)

class aslinear(linear_transform):
    """
    Return only linear part of affine transform
    """
    def __init__(self, transform):
        self._transform = astransform(transform)
        self.affine_offset = None
        self.input_shape = self._transform.output_shape
        self.output_shape = self._transform.input_shape

    def linear_map(self, x):
        return self._transform.linear_map(x)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        return self._transform.adjoint_map(x)

class adjoint(linear_transform):

    """
    Given an affine_transform, return a linear_transform
    that is the adjoint of its linear part.
    """
    def __init__(self, transform):
        self.transform = astransform(transform)
        self.affine_offset = None
        self.input_shape = self.transform.output_shape
        self.output_shape = self.transform.input_shape

    def linear_map(self, x):
        return self.transform.adjoint_map(x)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        return self.transform.linear_map(x)

class tensorize(affine_transform):

    """
    Given an affine_transform, return a linear_transform
    that expects q copies of something with transform's input_shape.

    This class effectively makes explicit that a transform
    may expect a matrix rather than a single vector.

    """
    def __init__(self, transform, q):
        self.transform = astransform(transform)
        self.affine_offset = self.transform.affine_offset
        self.input_shape = self.transform.input_shape + (q,)
        self.output_shape = self.transform.output_shape + (q,)

    def linear_map(self, x):
        return self.transform.linear_map(x)

    def affine_map(self, x):
        v = self.linear_map(x) 
        if self.affine_offset is not None:
            return v + self.affine_offset[:, np.newaxis]
        return v

    def adjoint_map(self, x):
        return self.transform.adjoint_map(x)

class residual(affine_transform):

    """
    Compute the residual from an affine transform.
    """

    def __init__(self, transform):
        self.transform = astransform(transform)
        self.input_shape = self.transform.input_shape
        self.output_shape = self.transform.output_shape
        self.affine_offset = None
        if not self.input_shape == self.output_shape:
            raise ValueError('output and input shapes should be the same to compute residual')

    def linear_map(self, x):
        return x - self.transform.linear_map(x)

    def affine_map(self, x):
        return x - self.transform.affine_map(x)

    def adjoint_map(self, u):
        return u - self.transform.adjoint_map(u)

class composition(affine_transform):

    """
    Composes a list of affine transforms, executing right to left
    """

    def __init__(self, *transforms):
        self.transforms = [astransform(t) for t in transforms]
        self.input_shape = self.transforms[-1].input_shape
        self.output_shape = self.transforms[0].output_shape

        # compute the affine_offset
        affine_offset = self.affine_map(np.zeros(self.input_shape))
        if not np.allclose(affine_offset, 0): 
            self.affine_offset = None
        else:
            self.affine_offset = affine_offset

    def linear_map(self, x):
        output = x
        for transform in self.transforms[::-1]:
            output = transform.linear_map(output)
        return output

    def affine_map(self, x):
        output = x
        for transform in self.transforms[::-1]:
            output = transform.affine_map(output)
        return output

    def adjoint_map(self, x):
        output = x
        for transform in self.transforms:
            output = transform.adjoint_map(output)
        return output

class affine_sum(affine_transform):

    """
    Creates the (weighted) sum of a list of affine_transforms
    """

    def __init__(self, transforms, weights=None):
        self.transforms = [astransform(T) for T in transforms]
        if weights is None:
            self.weights = np.ones(len(self.transforms))
        else:
            if not len(self.transforms) == len(weights):
                raise ValueError("Must specify a weight for each transform")
            self.weights = weights
        self.input_shape = self.transforms[0].input_shape
        self.output_shape = self.transforms[0].output_shape

        # compute the affine_offset
        affine_offset = self.affine_map(np.zeros(self.input_shape))
        if np.allclose(affine_offset, 0): 
            self.affine_offset = None
        else:
            self.affine_offset = affine_offset

    def linear_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.linear_map(x)
        return output

    def affine_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.affine_map(x)
        return output

    def adjoint_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.adjoint_map(x)
        return output


class scalar_multiply(affine_transform):

    def __init__(self, atransform, scalar):
        self.input_shape, self.output_shape = (atransform.input_shape, atransform.output_shape)
        self.scalar = scalar
        self.affine_offset = None
        self._atransform = atransform

    def affine_map(self, x):
        if self.scalar != 1.:
            return self._atransform.affine_map(x) * self.scalar
        else:
            return self._atransform.affine_map(x)

    def linear_map(self, x):
        if self.scalar != 1.:
            return self._atransform.linear_map(x) * self.scalar 
        else:
            return self._atransform.linear_map(x)


    def adjoint_map(self, x):
        if self.scalar != 1.:
            return self._atransform.adjoint_map(x) * self.scalar 
        else:
            return self._atransform.adjoint_map(x)

class posneg(affine_transform):

    def __init__(self, linear_transform):
        self.linear_transform = astransform(linear_transform)
        # where to store output so we don't recreate arrays 
        self.affine_offset = None
        self.input_shape = (2,) + self.linear_transform.input_shape
        self._adjoint_output = np.zeros(self.input_shape)
        self.output_shape = self.linear_transform.output_shape

    def linear_map(self, x):
        L = self.linear_transform.linear_map
        return  L(x[0]) - L(x[1])

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        u = self._adjoint_output
        u[0] = self.linear_transform.adjoint_map(x)
        u[1] = -u[0]
        return u

def todense(transform):
    """
    Return a dense array representation of a transform -- use
    carefully -- it could be large.
    """
    if len(transform.input_shape) == 1:
        I = np.identity(np.product(transform.input_shape))
        return transform.linear_map(I)
    else:
        raise NotImplementedError('expecting a 1D shape as input')

