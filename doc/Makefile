# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = #-q # suppress all output but warnings
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
BUILDDIR        = build
SRCDIR          = source
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SRCDIR)

# Required R packages
R_PACKAGES = Matrix gglasso

.PHONY: help clean pdf all dist public_html web web_public htmlonly api html pickle htmlhelp latex changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      	make HTML and API documents"
	@echo "  htmlonly  	make HTML documents only"
	@echo "  api       	make API documents only"
	@echo "  latex     	make LaTeX documents (you can set\
					PAPER=a4 or PAPER=letter)"
	@echo "  all		make HTML, API and PDF documents"
	@echo "  clean		remove all generated documents"
	@echo
	@echo "  linkcheck 	check all external links for integrity"
	@echo "  doctest   	run doctests in reST files"
	@echo "  pdf		make <latex> and run the PDF generation"
	@echo "  dist		make <all> and put results in dist/"
	@echo "  public_html	make <dist> and rsync with public website"
	@echo "  gitwash-update update git workflow from source repo"

# Commented these out, wasn't clear if we'd use these targets or not.
# @echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
# @echo "  htmlhelp  to make HTML files and a HTML help project"
# @echo "  changes   to make an overview over all changed/added/deprecated items"

clean:
	-rm -rf $(BUILDDIR)/* dist/* *~ api/generated
	-rm -f manual
	-rm -rf $(SRCDIR)/examples/*

pdf: latex
	cd $(BUILDDIR)/latex && make all-pdf

all: html pdf

dist: clean all
	mkdir -p dist
	ln $(BUILDDIR)/latex/nipy*.pdf dist/
	cp -al $(BUILDDIR)/html/* dist/
	@echo "Build finished.  Final docs are in dist/"

public_html: dist
	rsync -avH --delete dist/* $(WWW)

htmlonly:
	mkdir -p $(BUILDDIR)/html $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

api:
	-mkdir -p $(SRCDIR)/api/generated
	python ../tools/build_modref_templates.py
	@echo "Build API docs finished."

html: api htmlonly
	-ln -s $(BUILDDIR) manual
	@echo "Build HTML and API finished."

gitwash-update:
	python ../tools/gitwash_dumper.py devel/guidelines nipy

pickle:
	mkdir -p $(BUILDDIR)/pickle $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web $(BUILDDIR)/pickle"
	@echo "to start the sphinx-web server."

htmlhelp:
	mkdir -p $(BUILDDIR)/htmlhelp $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

latex: api
	mkdir -p $(BUILDDIR)/latex $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	mkdir -p $(BUILDDIR)/changes $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	mkdir -p $(BUILDDIR)/linkcheck $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	mkdir -p $(BUILDDIR)/doctest $(BUILDDIR)/doctrees
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo
	@echo "The overview file is in $(BUILDDIR)/doctest."

github: notebooks html
	# Needs ghp-import (pip install ghp-import)
	ghp-import -n -p $(BUILDDIR)/html/
	@echo
	@echo "Published to Github"

notebooks:
	python tools/check_r_packages.py $(R_PACKAGES)
	mkdir -p $(BUILDDIR)/html/notebooks
	python tools/convert_notebooks.py markdown $(SRCDIR)/examples --if-newer
	cp $(SRCDIR)/examples/*.ipynb $(BUILDDIR)/html/notebooks
