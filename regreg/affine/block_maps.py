import numpy as np

from . import (affine_transform, 
               adjoint,
               astransform)

class block_columns(affine_transform):

    """
    Blockwise transform: mapping where arguments 
    have common number of columns.

    Given a sequence of maps $X_i$ of shape
    $(n_i, q)$ produces an output of shape
    $(\sum_i n_i, q)$

    .. math::
    
       (X_1, \dots, X_k)(\beta_{p \times q) \mapsto 
       \begin{pmatrix} X_1\beta_1 \\ \dots X_k \beta_q 
       \end{pmatrix}

    """

    def __init__(self, transforms):

        '''
        Parameters
        ----------

        transforms : sequence
             A sequence of `affine_transform` whose outputs
             are similar in that their `input_shape` is common
             and their outputs are stackable vertically.

        '''

        self._transforms = [astransform(trans) for trans in transforms]
        if not np.all([trans.input_shape == self._transforms[0].input_shape 
                       for trans in self._transforms]):
            raise ValueError('each block should take same input shape')

        self.input_shape = self._transforms[0].input_shape + (len(self._transforms),)

        self._offsets = [trans.affine_map(np.zeros(trans.input_shape))
                         for trans in self._transforms]
        try:
            ndim = [o.ndim for o in self._offsets]
            if ndim[0] == 1:
                _buffer = np.hstack(self._offsets)
            elif ndim[0] == 2:
                _buffer = np.vstack(self._offsets)
        except ValueError:
            raise ValueError('output of transforms should be stackable')

        self.output_shape = _buffer.shape
        self._slices = []
        idx = 0
        for offset in self._offsets:
            self._slices.append(slice(idx, idx+offset.shape[0], 1))
            idx += offset.shape[0]

    def affine_map(self, arg):
        
        _buffer = np.empty(self.output_shape, np.float)
        for i, info in enumerate(zip(self._slices, 
                                     self._transforms)):
            slice, trans = info
            _buffer[slice] = trans.affine_map(arg[:,i])

        return _buffer

    def linear_map(self, arg):
        
        _buffer = np.empty(self.output_shape, np.float)
        for i, info in enumerate(zip(self._slices, 
                                     self._transforms)):
            slice, trans = info
            _buffer[slice] = trans.linear_map(arg[:,i])

        return _buffer

    def adjoint_map(self, arg):
        
        _adj_buffer = np.empty(self.input_shape, np.float)
        for i, info in enumerate(zip(self._slices, 
                                     self._transforms)):
            slice, trans = info
            _adj_buffer[:,i] = trans.adjoint_map(arg[slice])

        return _adj_buffer

class block_rows(affine_transform):

    """
    Blockwise transform: mapping where arguments 
    have common number of rows.

    Given a sequence of $k$ maps $X_i$ of shape
    $(p, n_i)$ produces an output of shape
    $(p, k)$.

    Formally this is an adjoint of `block_columns`

    .. math::
    
       (Z_1, \dots, Z_k)((Y_1, \dots, Y_k) \mapsto 
       \begin{pmatrix} Z_1 Y_1 & \dots & \dots Z_k Y_k \beta_q 
       \end{pmatrix}

    """

    def __init__(self, transforms):

        '''
        Parameters
        ----------

        transforms : sequence
             A sequence of `affine_transform` whose outputs
             are similar in that their `input_shape` is common
             and their outputs are stackable vertically.

        '''

        self._transforms = [astransform(trans) for trans in transforms]
        # we will use the linear map
        self._adjoint = block_columns([adjoint(trans) 
                                       for trans in self._transforms])
        self._adjointT = self._adjoint.T

        if not np.all([trans.output_shape == self._transforms[0].output_shape 
                       for trans in self._transforms]):
            raise ValueError('each block should produce same output shape')

        self.input_shape = (np.sum([trans.input_shape[0] for trans in self._transforms]),)
        self.output_shape = self._transforms[0].output_shape + (len(self._transforms),)

        _offsets = np.array([trans.affine_map(np.zeros(trans.input_shape))
                             for trans in self._transforms])
        try:
            ndim = [o.ndim for o in _offsets]
            if ndim[0] == 1:
                self._offset = _offsets.T
        except ValueError:
            raise ValueError('output of transforms should be the same and hstackable')

        self.output_shape = self._offset.shape

    def affine_map(self, arg):
        
        return self._adjointT.linear_map(arg) + self._offset

    def linear_map(self, arg):
        
        return self._adjointT.linear_map(arg) 

    def adjoint_map(self, arg):
        
        return self._adjoint.linear_map(arg)

