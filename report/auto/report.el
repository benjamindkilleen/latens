(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "INTERSPEECH2016"
    "graphicx"
    "amssymb"
    "amsmath"
    "bm"
    "bbm"
    "hyperref"
    "textcomp"
    "algorithm"
    "algpseudocode")
   (TeX-add-symbols
    "vec"
    "mat"
    "name")
   (LaTeX-add-labels
    "sec:introduction"
    "sec:method"
    "fig:overview"
    "alg:uniform-sampling"
    "sec:results"
    "fig:autoencoder-visualization"
    "sec:discussion")
   (LaTeX-add-bibliographies))
 :latex)

