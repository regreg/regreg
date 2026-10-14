""" Test doc templater
"""

import pytest
from regreg.doctemplates import (
    doc_templater,
    doc_template_provider,
    doc_template_user)


def test_basic():
    @doc_templater()
    class C(object):
        @doc_template_provider
        def f(self):
            "A docstring %(var)s"

    @doc_templater(dict(var='something'))
    class D(C):
        @doc_template_user
        def f(self):
            pass

    assert D().f.__doc__ == 'A docstring something'


def test_more_fully():
    @doc_templater()
    class C(object):
        @doc_template_provider
        def fa(self, arg):
            """A docstring %(a)s"""

        @doc_template_provider
        def fb(self, arg):
            """B docstring %(b)s"""

    # Providing a template sets the docstring to None
    assert C().fa.__doc__ == None

    @doc_templater(dict(a = 2, b='two'))
    class D(C):
        @doc_template_user
        def fa(self, arg):
            pass

        @doc_template_user
        def fb(self, arg):
            pass

    # Using a template
    assert D().fa.__doc__ == "A docstring 2"
    assert D().fb.__doc__ == "B docstring two"

    @doc_templater(dict(a = 3))
    class E(D):
        @doc_template_user
        def fa(self, arg):
            pass

        @doc_template_user
        def fb(self, arg):
            pass

    # Overriding a parameter in the template
    assert E().fa.__doc__ == "A docstring 3"
    assert E().fb.__doc__ == "B docstring two"

    def missing_param_error():
        @doc_templater(dict(a = 4))
        class Q(object):
            @doc_template_user
            def fb(self, arg):
                pass

    # Parameters inherited from subclass - missing - keyerror
    with pytest.raises(KeyError):
        missing_param_error()

    @doc_templater(dict(a = 4))
    class E(D):
        @doc_template_user
        @doc_template_provider
        def fa(self, arg):
            """Another docstring %(a)s"""

    # Can be both a user and a provider.
    assert E().fa.__doc__ == "Another docstring 4"
    assert E().fb.__doc__ == "B docstring two"

    @doc_templater(dict(a = 4))
    class F(D):
        @doc_template_provider
        @doc_template_user
        def fa(self, arg):
            """Another docstring %(a)s"""

    # The decorators can be in either order
    assert F().fa.__doc__ == "Another docstring 4"
    assert F().fb.__doc__ == "B docstring two"

    def null_doc_error(err=True):
        @doc_templater(dict(a = 99), doc_error=err)
        class Q(F):
            @doc_template_user
            def fa(self, arg):
                "A docstring too far"
        return Q

    # If docstring to be replaced is not empty, raise error by default
    with pytest.raises(ValueError):
        null_doc_error(True)
    assert null_doc_error(False)().fa.__doc__ == "Another docstring 99"

    @doc_templater(dict(a = 5))
    class G(F):
        @doc_template_user
        def fa(self, arg):
            pass

    # Showing the provision from before
    assert G().fa.__doc__ == "Another docstring 5"
    assert G().fb.__doc__ == "B docstring two"

    @doc_templater(dict(a=6, b='three'))
    class H(F):
        @doc_template_provider
        def fa(self, arg):
            """Yet another docstring %(a)s"""

    # Providing without using, again
    assert H().fa.__doc__ == None

    @doc_templater()
    class I(H):
        @doc_template_user
        def fa(self, arg):
            pass

        @doc_template_user
        def fb(self, arg):
            pass

    # Providing without using, again
    assert I().fa.__doc__ == "Yet another docstring 6"
    assert I().fb.__doc__ == "B docstring three"
