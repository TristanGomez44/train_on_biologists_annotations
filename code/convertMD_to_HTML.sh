pandoc -f markdown -t revealjs --slide-level=2 --section-divs --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js \
	--css=../css/style.css --highlight-style=zenburn -V revealjs-url=../../reveal.js/ --standalone  ../documents/pres_unsupObjDet.md -o ../documents/pres_unsupObjDet.html \
	--filter pandoc-citeproc --bibliography=../documents/biblio.bib 

firefox ../documents/pres_unsupObjDet.html
