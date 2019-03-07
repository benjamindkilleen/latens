(TeX-add-style-hook
 "template"
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
    "hyperref"
    "textcomp")
   (TeX-add-symbols
    "vec"
    "mat"
    "name")
   (LaTeX-add-labels
    "tab:example"
    "eq1"
    "eq2"
    "eq3"
    "eq4"
    "fig:speech_production")
   (LaTeX-add-bibliographies
    "report"))
 :latex)

