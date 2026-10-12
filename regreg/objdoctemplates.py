"""
Implement doc templates using objective templates and vars.
"""

from .doctemplates import doc_templater


def objective_doc_templater(doc_dict=None, doc_errors=True):
    """
    Return ``doc_templater`` class decorator for classes with objective docs.

    Check for specific case where class being decorated has attributes
    ``objective_template`` and ``objective_vars``.  Set new entry
    ``objective`` into class ``doc_dict`` using these class attributes, and
    copy each entry of ``objective_vars`` into ``doc_dict``.  Then continue
    decorating as for ``doc_templater``.

    Parameters
    ----------
    doc_dict : dict, optional
        Extra key / value pairs used to fill template docstrings.
    doc_errors : bool, optional
        Passed as `doc_error` to ``doc_templater``.

    Returns
    -------
    obj_kdec : function
        Class decorator.
    """
    if doc_dict is None:
        doc_dict = {}
    def obj_kdec(klass):
        if (hasattr(klass, 'objective_template') and
            hasattr(klass, 'objective_vars')):
            doc_dict['objective'] = (klass.objective_template %
                                     klass.objective_vars)
            for k, v in klass.objective_vars.items():
                doc_dict[k] = v
        return doc_templater(doc_dict, doc_errors)(klass)
    return obj_kdec

