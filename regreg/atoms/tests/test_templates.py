"""
Regression tests for atom reprs, objective templates and docstrings
(gh-64, gh-67).
"""
import importlib
import inspect
import pkgutil
import re

import numpy as np
import pytest

import regreg.api as rr
import regreg.atoms as A
from regreg.atoms import cones as C
from regreg.atoms.simplex import maximum
from regreg.atoms.sparse_group_lasso import sparse_group_lasso_dual

@pytest.mark.parametrize('make_atom', [
    lambda q: rr.sparse_group_lasso([1, 1, 2, 2, 2], [0, .5, .2, .2, .2], lagrange=1., quadratic=q),
    lambda q: rr.sparse_group_lasso([1, 1, 2, 2, 2], [0, .5, .2, .2, .2], bound=1., quadratic=q),
    lambda q: rr.slope([3., 2., 1.], lagrange=1., quadratic=q),
    lambda q: rr.slope([3., 2., 1.], bound=1., quadratic=q),
    lambda q: rr.weighted_l1norm([1., 2., 3.], lagrange=1., quadratic=q),
    lambda q: rr.weighted_supnorm([1., 2., 3.], bound=1., quadratic=q),
    lambda q: rr.l1norm(3, lagrange=1., quadratic=q),
    ])
def test_repr_quadratic(make_atom):
    # gh-64: the repr showed a zero quadratic and hid a nonzero one
    assert 'quadratic' not in repr(make_atom(None))
    assert 'quadratic' in repr(make_atom(rr.identity_quadratic(1., 0, 0, 0)))

def test_sparse_group_lasso_dual_base_conjugate():
    # gh-64: base_conjugate was rebuilt on every access
    d = sparse_group_lasso_dual([1, 1, 2, 2, 2], [0, .5, .2, .2, .2], bound=1.)
    x = np.array([1., -2, 3, 0.5, -1])
    assert d.base_conjugate is d.base_conjugate
    # the dual norm of x is about 1.6
    assert 1 < d.seminorm(x, lagrange=1.) < 2
    assert d.constraint(x) == np.inf
    # the cached atom does not depend on the bound ...
    d.bound = 2.
    assert d.constraint(x) == 0
    # ... and is rebuilt when the weights change
    conj = d.base_conjugate
    d.lasso_weights = d.lasso_weights * 2
    assert d.base_conjugate is not conj
    np.testing.assert_allclose(d.base_conjugate.lasso_weights, d.lasso_weights)

def test_sparse_group_lasso_dual_constraint():
    # constraint checked the unit dual ball whatever the bound, and returned
    # a bool, which nonsmooth_objective added to the value: +1 for feasible
    # points and +0 for infeasible ones
    x = np.array([1., -2, 3, 0.5, -1])
    norm = sparse_group_lasso_dual([1, 1, 2, 2, 2], [0, .5, .2, .2, .2],
                                   bound=1.).seminorm(x, lagrange=1.)
    for bound in [0.5 * norm, 0.9 * norm, 1.1 * norm, 3 * norm]:
        d = sparse_group_lasso_dual([1, 1, 2, 2, 2], [0, .5, .2, .2, .2], bound=bound)
        expected = 0 if norm <= bound else np.inf
        assert d.constraint(x) == expected
        assert d.nonsmooth_objective(x, check_feasibility=True) == expected

def test_objective_vars():
    # gh-67: l1_epigraph had its cone classes swapped, and maximum's
    # objective_vars were overwritten with simplex_constraint's
    assert C.l1_epigraph.objective_vars['coneklass'] == 'l1_epigraph'
    assert C.l1_epigraph.objective_vars['dualconeklass'] == 'l1_epigraph_polar'
    assert maximum.objective_vars['klass'] == 'maximum'
    assert maximum.objective_vars['dualklass'] == 'simplex_constraint'
    assert maximum.__doc__ is not None
    # these named another class, so the examples in their docstrings
    # constructed the wrong atom
    assert rr.sparse_group_lasso_dual.objective_vars['normklass'] == 'sparse_group_lasso_dual'
    assert rr.slope_conjugate.objective_vars['normklass'] == 'slope_conjugate'
    assert rr.weighted_l1norm.objective_vars['normklass'] == 'weighted_l1norm'
    assert rr.weighted_supnorm.objective_vars['normklass'] == 'weighted_supnorm'

def _atom_classes():
    seen = []
    for m in pkgutil.iter_modules(A.__path__):
        if m.name.startswith('_') or m.name == 'tests':
            continue
        mod = importlib.import_module('regreg.atoms.' + m.name)
        for _, k in inspect.getmembers(mod, inspect.isclass):
            if (k.__module__ == mod.__name__ and 'objective_template' in vars(k)
                and k not in seen):
                seen.append(k)
    return seen

@pytest.mark.parametrize('klass', _atom_classes(), ids=lambda k: k.__module__.split('.')[-1] + '.' + k.__name__)
def test_objective_template_latex(klass):
    # gh-67: unbalanced delimiters (\), \}}) and \in instead of \leq
    tex = klass.objective_template % dict(klass.objective_vars, var='x')
    assert '\\)' not in tex and '\\(' not in tex
    assert not re.search(r'\\in\s*-', tex)     # \in where \leq was meant
    t = tex.replace('\\left(', '(').replace('\\right)', ')')
    t = re.sub(r'\[[^\[\]()]*,[^\[\]()]*\)', '', t)  # half-open intervals
    t = t.replace('\\{', '').replace('\\}', '')
    for open_, close in ['()', '{}', '[]']:
        depth = 0
        for ch in t:
            depth += (ch == open_) - (ch == close)
            assert depth >= 0, tex
        assert depth == 0, tex

def test_group_lasso_epigraph_templates():
    # the dual epigraph template described var[1:] and var[0], but the code
    # uses arg[:-1] and arg[-1]
    for klass in [rr.group_lasso_epigraph, rr.group_lasso_epigraph_polar,
                  rr.group_lasso_dual_epigraph, rr.group_lasso_dual_epigraph_polar]:
        tex = klass.objective_template % dict(klass.objective_vars, var='x')
        assert 'x[-1]' in tex and 'x[0]' not in tex and 'x[1:]' not in tex

def test_cone_classmethod_docstrings():
    # gh-67: doc_template_provider cleared the docstrings of cone.linear and
    # cone.affine
    for f in [C.cone.linear, C.cone.affine, rr.nonnegative.linear, rr.nonnegative.affine]:
        assert f.__doc__ is not None and 'Compose a cone constraint' in f.__doc__
