.. _release-guide:

**********************************
A guide to making a Regreg release
**********************************

This is a guide for developers who are doing a Regreg release.

The general idea of these instructions is to go through the following steps:

* Make sure that the code is in the right state for release;
* update release-related docs such as the Changelog;
* update various documents giving dependencies, dates and so on;
* check all standard and release-specific tests pass;
* make the *release commit* and release tag;
* check Windows binary builds and slow / big memory tests;
* push source and windows builds to pypi;
* push docs;
* push release commit and tag to github;
* announce.

We leave pushing the tag to the last possible moment, because it's very bad
practice to change a Git tag once it has reached the public servers (in our
case, Github).  So we want to make sure of the contents of the release before
pushing the tag.

.. _release-checklist:

Release checklist
=================

* Review the open list of `regreg issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Review and update the release notes.  Review and update the ``Changelog``
  file.  Get a partial list of contributors with something like::

      git log 0.1.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``0.1.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.1.0..`` to make sure the release
  notes are as complete as possible and that every contributor was recognized.

* Look at ``doc/source/index.rst`` and add any authors not yet acknowledged.
  You might want to use the following to list authors by the date of their
  contributions::

    git log --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'

  (From:
  http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit#6482473)

  Consider any updates to the ``AUTHOR`` file.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -nse``.

* Check the copyright year in ``doc/source/conf.py`` and ``LICENSE``.

* Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by pypi_.  ``rst2html.py`` is a script
  installed by docutils_ (``pip install docutils``).

* Check the dependencies listed in ``regreg/info.py`` (e.g.
  ``NUMPY_MIN_VERSION``) and in ``doc/source/installation.rst`` and in
  ``requirements.txt`` and in ``.travis.yml``.  They should at least match. Do
  they still hold?  Make sure `regreg on travis`_ is testing the minimum
  dependencies specifically.

* Do a final check on the `nipy buildbot`_.  Use the ``try_branch.py``
  scheduler available in nibotmi_ to test particular schedulers.

* Make sure all tests pass (from the Regreg root directory)::

    nosetests --with-doctest regreg

* Make sure you are set up to use the ``try_branch.py`` - see
  https://github.com/nipy/nibotmi/blob/master/install.rst#trying-a-set-of-changes-on-the-buildbots

* Make sure all your changes are committed or removed, because
  ``try_branch.py`` pushes up the changes in the working tree;

  * Check the documentation doctests::

      cd doc && make clean && make doctest

    This should also be tested by `regreg on travis`_.

  * Check everything compiles without syntax errors::

      python -m compileall .

* Check on different platforms, particularly windows and PPC. Look at the
  `nipy buildbot`_ automated test runs for this;

* Build and test the Regreg wheels.  See the `wheel builder README
  <https://github.com/MacPython/regreg-wheels>`_ for instructions.  In
  summary, clone the wheel-building repo, edit the ``.travis.yml`` text files
  with the branch or commit for the release, commit and then push back up to
  github.  This will trigger a wheel build and test on OSX and Linux. Check
  the build has passed on on the Travis-CI interface at
  https://travis-ci.org/MacPython/regreg-wheels.  You'll need commit
  privileges to the ``regreg-wheels`` repo.

* Make sure you have travis-ci_ building set up for your own repo. Make a new
  ``release-check`` (or similar) branch, and push the code in its current
  state to a branch that will build, e.g::

    git branch -D release-check # in case branch already exists
    git co -b release-check
    # You might need the --force flag here
    git push your-github-user release-check -u

* Once everything looks good, you are ready to upload the source release to
  PyPi.  See `setuptools intro`_.  Make sure you have a file
  ``\$HOME/.pypirc``, of form::

    [distutils]
    index-servers =
        pypi
        warehouse

    [pypi]
    username:your.pypi.username
    password:your-password

    [warehouse]
    repository: https://upload.pypi.io/legacy/
    username:your.pypi.username
    password:your-password

* Clean::

    make distclean
    # Check no files outside version control that you want to keep
    git status
    # Nuke
    git clean -fxd

* When ready::

    python setup.py register
    python setup.py sdist --formats=zip
    # -s flag to sign the release
    twine upload -r warehouse -s dist/regreg*

* Tag the release with signed tag of form ``0.1.0``::

    git tag -s 0.1.0

* Push the tag and any other changes to trunk with::

    git push origin 0.1.0
    git push

* Now the version number is OK, push the docs to github pages with::

    make upload-html

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/0.1.x

    Commit.  Don't forget to push upstream with something like::

      git push upstream-remote maint/0.1.x --set-upstream

  * Start next development series::

      git co main-master

* Push the main branch::

    git push upstream-remote main-master

* Announce to the mailing lists.

.. _setuptools intro: https://pythonhosted.org/an_example_pypi_project/setuptools.html

.. include:: ../links_names.inc
