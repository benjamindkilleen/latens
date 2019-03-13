(TeX-add-style-hook
 "presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "10pt" "usenames" "dvipsnames" "table")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "graphicx"
    "array"
    "fix-cm"
    "colortbl"
    "hyperref"
    "graphbox"
    "algorithm"
    "algpseudocode")
   (TeX-add-symbols
    "kl")
   (LaTeX-add-labels
    "sec:motivation"
    "fig:supervised"
    "fig:imagenet"
    "fig:unsupervised"
    "fig:semi-supervised"
    "fig:scientific-images"
    "sec:method"
    "fig:overview"
    "sec:results"
    "fig:conv-encodings"
    "fig:conv-uniform"
    "sec:references")
   (LaTeX-add-bibliographies))
 :latex)

