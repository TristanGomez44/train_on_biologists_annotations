pandoc -f markdown -t revealjs --slide-level=2 --section-divs --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js \
	--css=../css/style.css --highlight-style=zenburn -V revealjs-url=../../reveal.js/ --standalone  ../documents/pres_18-12-2019.md -o ../documents/pres_18-12-2019.html \
	--filter pandoc-citeproc --bibliography=../documents/biblio.bib -V transition=none

firefox ../documents/pres_18-12-2019.html
